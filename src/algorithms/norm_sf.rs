use pyo3::prelude::*;

pub trait NormSf<T: Clone + PartialOrd> {
    /// Survival function (1 - `cdf`) at x of the given RV.
    fn norm_sf(self) -> T;
}

impl NormSf<f32> for f32 {
    fn norm_sf(self) -> f32 {
        ((statrs::function::erf::erfc(self as f64 / 2.0f64.sqrt())) / 2.0) as f32
    }
}

impl NormSf<f64> for f64 {
    fn norm_sf(self) -> f64 {
        (statrs::function::erf::erfc(self / 2.0f64.sqrt())) / 2.0
    }
}

/// Survival function (1 - `cdf`) at x of the given RV.
pub fn norm_sf(x: f64) -> f64 {
    (statrs::function::erf::erfc(x / 2.0f64.sqrt())) / 2.0
}

#[pyfunction]
#[pyo3(name = "norm_sf")]
/// Survival function (1 - `cdf`) at x of the given RV. Same as `scipy.stats.norm.sf(x)`.
pub fn norm_sf_py(x: f64) -> f64 {
    norm_sf(x)
}
