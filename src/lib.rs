mod lattice;
mod tempering;

use crate::lattice::Lattice;
use crate::tempering::LatticeTempering;
use pyo3::prelude::*;

#[pymodule]
fn py_monte_carlo(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Lattice>()?;
    m.add_class::<LatticeTempering>()?;
    Ok(())
}
