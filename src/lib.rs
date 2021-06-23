mod classicising;
mod lattice;
mod qmcising;
mod qmcrunner;
mod tempering;

use crate::classicising::ClassicIsing;
use crate::lattice::Lattice;
use crate::qmcising::QmcIsing;
use crate::qmcrunner::QmcRunner;
use crate::tempering::LatticeTempering;
use pyo3::prelude::*;

#[pymodule]
fn py_monte_carlo(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Lattice>()?;
    m.add_class::<LatticeTempering>()?;
    m.add_class::<QmcRunner>()?;
    m.add_class::<QmcIsing>()?;
    m.add_class::<ClassicIsing>()?;
    Ok(())
}
