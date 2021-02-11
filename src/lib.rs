mod lattice;
mod qmcrunner;
mod tempering;

use crate::lattice::Lattice;
use crate::qmcrunner::*;
use crate::tempering::LatticeTempering;
use pyo3::prelude::*;

#[pymodule]
fn py_monte_carlo(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Lattice>()?;
    m.add_class::<LatticeTempering>()?;
    m.add_class::<QmcRunner>()?;
    Ok(())
}
