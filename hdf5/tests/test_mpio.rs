//! Reading and writing through the MPI-IO driver with a [`DatasetTransfer`] property list.
//!
//! Runs as a single MPI process (no `mpirun` needed), which already goes through the parallel
//! code paths. HDF5 records on the transfer property list how it actually did each transfer,
//! which is what shows the property list reached the library.
//!
//! One test function only: MPI may be initialized once per process, and its collectives must not
//! run from two test threads at once.
#![cfg(feature = "mpio")]

use std::os::raw::c_int;
use std::ptr;

use hdf5_sys::h5d::H5D_mpio_actual_io_mode_t;
use hdf5_sys::h5i::hid_t;
use mpi_sys::{MPI_Init, MPI_Initialized, RSMPI_COMM_WORLD};

use hdf5::dataset::{DatasetTransfer, MpioTransferMode};
use hdf5_metno as hdf5;

fn mpio_file(dir: &tempfile::TempDir) -> hdf5::Result<hdf5::File> {
    let mut initialized: c_int = 1;
    unsafe { MPI_Initialized(&mut initialized) };
    if initialized == 0 {
        unsafe { MPI_Init(ptr::null_mut(), ptr::null_mut()) };
    }

    hdf5::File::with_options()
        .with_fapl(|p| p.mpio(unsafe { RSMPI_COMM_WORLD }, None))
        .create(dir.path().join("mpio.h5"))
}

// not bound by `hdf5-sys`; part of the library since 1.8.8
unsafe extern "C" {
    fn H5Pget_mpio_actual_io_mode(
        plist_id: hid_t, actual_io_mode: *mut H5D_mpio_actual_io_mode_t,
    ) -> hdf5_sys::h5::herr_t;
}

/// How HDF5 actually did the last transfer made with `plist`.
fn actual_io_mode(plist: &DatasetTransfer) -> H5D_mpio_actual_io_mode_t {
    let mut mode = H5D_mpio_actual_io_mode_t::H5D_MPIO_NO_COLLECTIVE;
    assert!(unsafe { H5Pget_mpio_actual_io_mode(plist.id(), &mut mode) } >= 0);
    mode
}

fn transfer(mode: MpioTransferMode) -> hdf5::Result<DatasetTransfer> {
    DatasetTransfer::build().mpio(mode).finish()
}

#[test]
fn test_collective_transfer() -> hdf5::Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let file = mpio_file(&dir)?;

    write_a_whole_dataset(&file)?;
    if hdf5::filters::deflate_available() {
        write_a_filtered_dataset(&file)?;
    }

    Ok(())
}

fn write_a_filtered_dataset(file: &hdf5::File) -> hdf5::Result<()> {
    let data: Vec<f64> = (0..1000).map(f64::from).collect();
    let ds = file.new_dataset::<f64>().shape(1000).chunk(100).deflate(4).create("filtered")?;

    // into a dataset of its own: HDF5 1.10 fails a collective write over chunks an independent
    // one already wrote (`MPI_File_set_view`: invalid datatype)
    let independent = transfer(MpioTransferMode::Independent)?;
    let other = file.new_dataset::<f64>().shape(1000).chunk(100).deflate(4).create("other")?;
    other.as_writer().transfer(&independent).write_slice(&data[..500], 0..500)?;
    assert_eq!(actual_io_mode(&independent), H5D_mpio_actual_io_mode_t::H5D_MPIO_NO_COLLECTIVE);

    let collective = transfer(MpioTransferMode::Collective)?;
    ds.as_writer().transfer(&collective).write_slice(&data[..500], 0..500)?;
    assert_eq!(actual_io_mode(&collective), H5D_mpio_actual_io_mode_t::H5D_MPIO_CHUNK_COLLECTIVE);
    ds.as_writer().transfer(&collective).write_slice(&data[500..], 500..1000)?;
    // a rank with nothing to write still takes part in a collective write
    ds.as_writer().transfer(&collective).write_slice(&data[..0], 0..0)?;

    let read: Vec<f64> = ds.as_reader().transfer(&collective).read_slice_1d(0..1000)?.to_vec();
    assert_eq!(read, data);
    let empty: Vec<f64> = ds.as_reader().transfer(&collective).read_slice_1d(0..0)?.to_vec();
    assert!(empty.is_empty());

    Ok(())
}

fn write_a_whole_dataset(file: &hdf5::File) -> hdf5::Result<()> {
    let data: Vec<i32> = (0..64).collect();
    let ds = file.new_dataset::<i32>().shape(64).create("whole")?;

    let collective = transfer(MpioTransferMode::Collective)?;
    ds.as_writer().transfer(&collective).write(&data)?;
    assert_eq!(
        actual_io_mode(&collective),
        H5D_mpio_actual_io_mode_t::H5D_MPIO_CONTIGUOUS_COLLECTIVE
    );
    assert_eq!(ds.as_reader().transfer(&collective).read_raw::<i32>()?, data);

    Ok(())
}
