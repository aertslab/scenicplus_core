use ndarray::{Array1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "get_nonzero_row_indices")]
/// Get the indices of the rows that have at least one nonzero element.
///
/// This function is equivalent to:
///     np.nonzero(np.count_nonzero(x, axis=1))[0]
/// Parameters
/// ----------
/// arr
///     2D float32 continuous numpy array.
///
/// Returns
/// -------
/// row indices of the rows that have at least one nonzero element.
pub fn get_nonzero_row_indices_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray1<usize>> {
    // Get a view of the array.
    let arr: ArrayView2<f32> = arr.as_array();
    let n_rows = arr.shape()[0];

    // Pre-allocate the result array.
    let mut nonzero_row_indices = Vec::with_capacity(n_rows);

    // Iterate through each row
    for (i, row) in arr.rows().into_iter().enumerate() {
        // Check if the row has any nonzero elements.
        if row.iter().any(|&val| val != 0.0) {
            nonzero_row_indices.push(i);
        }
    }

    // Convert the Vec to an ndarray Array1 and then to a PyArray.
    Array1::from_vec(nonzero_row_indices).into_pyarray(py)
}
