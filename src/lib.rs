use pyo3::prelude::*;
use pyo3::{pymodule, PyResult, Python};

mod algorithms;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn scenicplus_core(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let algorithms_submodule = PyModule::new(py, "algorithms")?;

    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_i8_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_i16_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_i32_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_i64_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_u8_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_u16_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_u32_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_u64_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_f32_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::arg_sort::arg_sort_f64_py,
        module
    )?)?;

    algorithms_submodule
        .add_function(wrap_pyfunction!(algorithms::norm_sf::norm_sf_py, module)?)?;

    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::rank_sums::rank_sums_py,
        module
    )?)?;
    algorithms_submodule.add_function(wrap_pyfunction!(
        algorithms::rank_sums::rank_sums_2d_py,
        module
    )?)?;

    module.add_submodule(&algorithms_submodule)?;

    Ok(())
}
