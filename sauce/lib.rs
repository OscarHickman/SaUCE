//! Compiled core of `pysauce`: pair and N-tuple counters for periodic boxes
//! whose particles are labelled by partition.
//!
//! * [`grid`] – periodic cell grid used to find neighbours.
//! * [`binning`] – histogram bin lookup.
//! * [`traverse`] – enumeration of the pairs within a given reach.
//! * [`pairs`] – pair counters.
//! * [`npoint`] – triplet and higher-order tuple counters.
//! * [`validate`] – input checks.
//! * [`python`] – the Python-facing functions.

use pyo3::prelude::*;

mod binning;
mod grid;
mod npoint;
mod pairs;
mod python;
mod traverse;
mod validate;

#[pymodule]
fn _pysauce(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::count_pairs_1d, m)?)?;
    m.add_function(wrap_pyfunction!(python::count_pairs_1d_partition_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(python::count_pairs_2d, m)?)?;
    m.add_function(wrap_pyfunction!(python::count_pairs_smu, m)?)?;
    m.add_function(wrap_pyfunction!(python::count_npoint, m)?)?;
    m.add_function(wrap_pyfunction!(python::count_npoint3_shape, m)?)?;
    m.add_function(wrap_pyfunction!(python::count_npoint3_smu, m)?)?;
    Ok(())
}
