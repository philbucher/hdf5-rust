//! Dataset transfer properties.

/*
Not implemented:
- H5P{set,get}_buffer, H5P{set,get}_type_conv_cb, H5P{set,get}_vlen_mem_manager,
  H5P{set,get}_data_transform, H5P{set,get}_hyper_vector_size, H5P{set,get}_btree_ratios,
  H5P{set,get}_edc_check, H5P{set,get}_filter_callback, H5Pset_preserve
- H5Pset_dxpl_mpio_{collective_opt,chunk_opt,chunk_opt_num,chunk_opt_ratio}
*/

use std::fmt::{self, Debug};
use std::ops::Deref;

use hdf5_sys::h5p::H5Pcreate;
#[cfg(feature = "have-parallel")]
use hdf5_sys::h5p::{H5FD_mpio_xfer_t, H5Pget_dxpl_mpio, H5Pset_dxpl_mpio};

use crate::globals::H5P_DATASET_XFER;
use crate::internal_prelude::*;

/// Dataset transfer properties.
#[repr(transparent)]
pub struct DatasetTransfer(Handle);

impl ObjectClass for DatasetTransfer {
    const NAME: &'static str = "dataset transfer property list";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_GENPROP_LST];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.is_class(PropertyListClass::DataTransfer),
            "expected dataset transfer property list, got {:?}",
            self.class()
        );
        Ok(())
    }
}

impl Debug for DatasetTransfer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = f.debug_struct("DatasetTransfer");
        #[cfg(feature = "have-parallel")]
        formatter.field("mpio", &self.mpio());
        formatter.finish()
    }
}

impl Deref for DatasetTransfer {
    type Target = PropertyList;

    fn deref(&self) -> &PropertyList {
        unsafe { self.transmute() }
    }
}

impl PartialEq for DatasetTransfer {
    fn eq(&self, other: &Self) -> bool {
        <PropertyList as PartialEq>::eq(self, other)
    }
}

impl Eq for DatasetTransfer {}

impl Clone for DatasetTransfer {
    fn clone(&self) -> Self {
        unsafe { self.deref().clone().cast_unchecked() }
    }
}

/// How the MPI-IO driver transfers data: every rank on its own, or all ranks together.
///
/// Writing a filtered (e.g. compressed) dataset from more than one rank requires
/// [`Collective`](Self::Collective). Every rank must then take part in each read or write, a rank
/// with nothing to transfer included.
#[cfg(feature = "have-parallel")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MpioTransferMode {
    /// Every rank transfers its data independently of the others (HDF5's default).
    #[default]
    Independent,
    /// All ranks transfer their data in one collective operation.
    Collective,
}

#[cfg(feature = "have-parallel")]
impl From<H5FD_mpio_xfer_t> for MpioTransferMode {
    fn from(mode: H5FD_mpio_xfer_t) -> Self {
        match mode {
            H5FD_mpio_xfer_t::H5FD_MPIO_COLLECTIVE => Self::Collective,
            H5FD_mpio_xfer_t::H5FD_MPIO_INDEPENDENT => Self::Independent,
        }
    }
}

#[cfg(feature = "have-parallel")]
impl From<MpioTransferMode> for H5FD_mpio_xfer_t {
    fn from(mode: MpioTransferMode) -> Self {
        match mode {
            MpioTransferMode::Collective => Self::H5FD_MPIO_COLLECTIVE,
            MpioTransferMode::Independent => Self::H5FD_MPIO_INDEPENDENT,
        }
    }
}

/// Builder used to create dataset transfer property list.
#[derive(Clone, Debug, Default)]
pub struct DatasetTransferBuilder {
    #[cfg(feature = "have-parallel")]
    mpio: Option<MpioTransferMode>,
}

impl DatasetTransferBuilder {
    /// Creates a new dataset transfer property list builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder from an existing property list.
    pub fn from_plist(plist: &DatasetTransfer) -> Result<Self> {
        let mut builder = Self::default();
        #[cfg(feature = "have-parallel")]
        builder.mpio(plist.get_mpio()?);
        #[cfg(not(feature = "have-parallel"))]
        let _ = plist;
        Ok(builder)
    }

    /// Sets the MPI-IO data transfer mode.
    #[cfg(feature = "have-parallel")]
    pub fn mpio(&mut self, mode: MpioTransferMode) -> &mut Self {
        self.mpio = Some(mode);
        self
    }

    fn populate_plist(&self, id: hid_t) -> Result<()> {
        #[cfg(feature = "have-parallel")]
        if let Some(v) = self.mpio {
            h5try!(H5Pset_dxpl_mpio(id, v.into()));
        }
        #[cfg(not(feature = "have-parallel"))]
        let _ = id;
        Ok(())
    }

    /// Copies the builder settings into a dataset transfer property list.
    pub fn apply(&self, plist: &mut DatasetTransfer) -> Result<()> {
        h5lock!(self.populate_plist(plist.id()))
    }

    /// Constructs a new dataset transfer property list.
    pub fn finish(&self) -> Result<DatasetTransfer> {
        h5lock!({
            let mut plist = DatasetTransfer::try_new()?;
            self.apply(&mut plist).map(|()| plist)
        })
    }
}

/// Dataset transfer property list.
impl DatasetTransfer {
    /// Creates a new dataset transfer property list.
    pub fn try_new() -> Result<Self> {
        Self::from_id(h5try!(H5Pcreate(*H5P_DATASET_XFER)))
    }

    /// Creates a copy of the property list.
    pub fn copy(&self) -> Self {
        unsafe { self.deref().copy().cast_unchecked() }
    }

    /// Creates a new dataset transfer property list builder.
    pub fn build() -> DatasetTransferBuilder {
        DatasetTransferBuilder::new()
    }

    #[cfg(feature = "have-parallel")]
    #[doc(hidden)]
    pub fn get_mpio(&self) -> Result<MpioTransferMode> {
        // not through `h5get!`, which needs a `Default` the raw enum does not implement
        let mut mode = H5FD_mpio_xfer_t::H5FD_MPIO_INDEPENDENT;
        h5try!(H5Pget_dxpl_mpio(self.id(), &mut mode));
        Ok(mode.into())
    }

    /// Returns the MPI-IO data transfer mode.
    #[cfg(feature = "have-parallel")]
    pub fn mpio(&self) -> MpioTransferMode {
        self.get_mpio().unwrap_or_default()
    }
}
