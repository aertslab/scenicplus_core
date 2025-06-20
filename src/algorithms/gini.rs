use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use voracious_radix_sort::RadixSort;

#[pyfunction]
#[pyo3(name = "gini")]
/// Calculate the Gini coefficient for 1D numpy array.
///
/// The Gini coefficient is a measure of statistical dispersion that represents
/// the income or wealth distribution of a nation's residents, and is commonly used
/// as a measure of inequality.
///
/// After sorting the array, the Gini coefficient is calculated as:
///     G = \frac{\sum_{i=1}^{n} (2i - n - 1) x_i}{n \sum_{i=1}^{n} x_i}
/// where:
/// - `n` is the number of elements in the array,
/// - `x_i` is the value at index `i` in the sorted array.
///
/// Parameters
/// ----------
/// arr
///     1D float64 continuous numpy array with all values greater than zero.
///
/// Returns
/// -------
/// gini_coefficient
///     The Gini coefficient, a value between 0 and 1, where 0 represents perfect
///     equality and 1 represents perfect inequality.
///
pub fn gini_py<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
    // Convert numpy 1D array to vector.
    let mut arr = arr.as_array().to_owned().into_raw_vec();

    // Sort array inplace.
    arr.voracious_sort();

    // Get number of elements in the array.
    let n = arr.len();

    // Check if all values are greater than zero.
    // As the array is sorted, we only need to check the first element.
    if n >= 1 {
        if arr[0] <= 0.0 {
            return Err(PyValueError::new_err("Values should be greater than 0.0."));
        }
    }

    // Get the sum of the array.
    let sum_array: f64 = arr.iter().sum();

    // If sum is zero, return 0 (to avoid division by zero).
    if sum_array == 0.0 {
        return Ok(0.0);
    }

    // Calculate (2 * index - n - 1) for each element.
    let weighted_indices = (1..=n).map(|i| 2.0 * i as f64 - n as f64 - 1.0);

    // Calculate the Gini coefficient
    let gini_coefficient = weighted_indices
        .zip(arr.iter())
        .map(|(wi, &x)| wi as f64 * x)
        .sum::<f64>()
        / (n as f64 * sum_array);

    Ok(gini_coefficient)
}
