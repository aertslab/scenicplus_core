use pyo3::prelude::*;
use pyo3::{pymodule, PyResult, Python};

mod algorithms;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn scenicplus_core(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let algorithms_module = PyModule::new(py, "algorithms")?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_i8_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_i16_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_i32_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_i64_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_u8_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_u16_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_u32_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_u64_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_f32_py, m)?)?;
    algorithms_module.add_function(wrap_pyfunction!(algorithms::arg_sort::arg_sort_f64_py, m)?)?;
    m.add_submodule(algorithms_module)?;

    Ok(())
}
