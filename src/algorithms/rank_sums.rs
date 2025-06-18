use pyo3::prelude::*;
use rayon::prelude::*;

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::Python;

use crate::algorithms::{arg_sort::ArgSort, norm_sf::NormSf};

use voracious_radix_sort::RadixSort;

/// Assign ranks to data, where for tied values the average ranking that
/// would have been assigned to all the tied values is assigned to each value
/// in the group. Return the sum of the ranks at position `idx`.
///
/// Slices which contain NaNs are not supported.
///
/// Loosely based on `scipy.stats.rankdata`.
fn rank_data_avg_at_idx(arr: &[u32], idx: usize) -> f64 {
    let len = arr.len();
    let sort_idx = arr.arg_sort_fastest();
    let mut count = 0;
    let mut inv = Vec::with_capacity(len);
    unsafe {
        inv.set_len(len);
    }
    sort_idx.iter().for_each(|&i| {
        *inv.get_mut(i).unwrap() = count;
        count += 1;
    });

    let sorted = unsafe { sort_idx.iter().map(|i| *arr.get_unchecked(*i)) };

    let not_consecutive_same_slice1 = sorted.clone().skip(1);
    let not_consecutive_same_slice2 = sorted.take(len - 1);

    let not_consecutive_same = not_consecutive_same_slice1
        .zip(not_consecutive_same_slice2)
        .map(|(x, y)| x != y);

    let mut count = Vec::with_capacity(len + 1);
    let mut dense_tmp = Vec::with_capacity(len);

    let mut cnt = 0;
    let mut cumsum = 1;
    count.push(cnt);
    dense_tmp.push(cumsum);

    not_consecutive_same.for_each(|b| {
        cnt += 1;
        if b {
            count.push(cnt);

            cumsum += 1;
        }
        dense_tmp.push(cumsum)
    });
    count.push(len);

    let dense = inv.iter().map(|i| *dense_tmp.get(*i).unwrap());

    let rank_sums_avg = dense.map(|i| {
        let x = unsafe { *count.get_unchecked(i) } as f64;
        let y = unsafe { *count.get_unchecked(i - 1) } as f64 + 1.0;
        (x + y) * 0.5
    });
    rank_sums_avg.take(idx).sum()
}

/// Compute the Wilcoxon rank-sum statistic for two samples.
///
/// The Wilcoxon rank-sum test tests the null hypothesis that two sets
/// of measurements are drawn from the same distribution.  The alternative
/// hypothesis is that values in one sample are more likely to be
/// larger than the values in the other sample.
///
/// This test should be used to compare two samples from continuous
/// distributions.  It does not handle ties between measurements
/// in x and y and does not support NaNs.
///
/// Based on `scipy.stats.ranksums` implementation.
fn rank_sums(x: &[u32], y: &[u32]) -> (f64, f64) {
    let n1 = x.len();
    let n2 = y.len();

    // Allocate a single vector to hold concatenated x and y arrays.
    let mut all_data = Vec::with_capacity(n1 + n2);
    // Set length of uninitialized vector as it will be filled in the next step with the actual data.
    unsafe {
        all_data.set_len(n1 + n2);
    }
    // Populate the vector with the contents of x and y.
    all_data[..n1].clone_from_slice(x);
    all_data[n1..].clone_from_slice(y);

    // Sort the y part of `all_data` inplace as later `arg_sort` (in `rank_data_avg_at_idx`)
    // will  be ran on `all_data`, but only for the x part of the array the index order
    // needs to be preserved. Running sort on the y part of the data and arg_sort on the
    // full data is much faster than running `arg_sort``without sorting the y part first.
    let y_sorted = &mut all_data[n1..];
    y_sorted.voracious_sort();

    // Compute the rank sums using the average method for the x and y arrays.
    let s = rank_data_avg_at_idx(&all_data, n1);
    let n1 = n1 as f64;
    let n2 = n2 as f64;
    let expected = n1 * (n1 + n2 + 1.0) / 2.0;
    let z = (s - expected) / f64::sqrt(n1 * n2 * (n1 + n2 + 1.0) / 12.0);
    let prob = 2.0 * z.abs().norm_sf();
    (z, prob)
}

#[pyfunction]
#[pyo3(name = "rank_sums")]
/// Compute the Wilcoxon rank-sum statistic for two samples.
///
/// The Wilcoxon rank-sum test tests the null hypothesis that two sets
/// of measurements are drawn from the same distribution.  The alternative
/// hypothesis is that values in one sample are more likely to be
/// larger than the values in the other sample.
///
/// This test should be used to compare two samples from continuous
/// distributions.  It does not handle ties between measurements
/// in x and y and does not support NaNs.
///
/// Based on `scipy.stats.ranksums` implementation.
pub fn rank_sums_py<'py>(
    _py: Python<'py>,
    x: PyReadonlyArray1<'_, u32>,
    y: PyReadonlyArray1<'_, u32>,
) -> (f64, f64) {
    let x = x.as_slice().expect("input not contiguous");
    let y = y.as_slice().expect("input not contiguous");

    let (z, p) = rank_sums(x, y);
    (z, p)
}

#[pyfunction]
#[pyo3(name = "rank_sums_2d")]
/// Compute the Wilcoxon rank-sum statistic for two samples for each row in x and y.
///
/// The Wilcoxon rank-sum test tests the null hypothesis that two sets
/// of measurements are drawn from the same distribution.  The alternative
/// hypothesis is that values in one sample are more likely to be
/// larger than the values in the other sample.
///
/// This test should be used to compare two samples from continuous
/// distributions.  It does not handle ties between measurements
/// in x and y and does not support NaNs.
///
/// Based on `scipy.stats.ranksums` implementation.
pub fn rank_sums_2d_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, u32>,
    y: PyReadonlyArray2<'_, u32>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    // Get the number of rows in the input arrays.
    let n_rows = x.shape()[0];
    let n_rows_y = y.shape()[0];

    assert_eq!(n_rows, n_rows_y);

    // Preallocate vectors to store the results.
    let mut z = Vec::with_capacity(n_rows);
    let mut p = Vec::with_capacity(n_rows);

    // Convert Numpy arrays to ndarrays.
    let x_array = x.as_array();
    let y_array = y.as_array();

    // Use rayon's par_iter to parallelize rank sums computation across rows.
    (0..n_rows)
        .into_par_iter()
        .map(|i| {
            // Create bindings to extend the lifetime of the row references.
            let x_row = x_array.row(i);
            let y_row = y_array.row(i);

            // Now get the slices from the longer-lived bindings.
            let x_slice = x_row.as_slice().unwrap();
            let y_slice = y_row.as_slice().unwrap();

            // Compute the rank sums for the current row.
            rank_sums(x_slice, y_slice)
        })
        .unzip_into_vecs(&mut z, &mut p);

    // Convert directly to PyArrays.
    (
        Array1::from_vec(z).into_pyarray(py),
        Array1::from_vec(p).into_pyarray(py),
    )
}
