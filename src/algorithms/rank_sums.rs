use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::Python;

use crate::algorithms::{arg_sort::ArgSort, norm_sf::NormSf};

/// Assign ranks to data, where for tied values the average ranking that
/// would have been assigned to all the tied values is assigned to each value.
///
/// Slices which contain NaNs are not support
///
/// Loosely based on `scipy.stats.rankdata`.
fn rank_data_avg(arr: &[i64]) -> Vec<f64> {
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

    let mut rank_sums_avg: Vec<f64> = Vec::with_capacity(dense.len());
    rank_sums_avg.extend(dense.map(|i| {
        let x = unsafe { *count.get_unchecked(i) } as f64;
        let y = unsafe { *count.get_unchecked(i - 1) } as f64 + 1.0;
        (x + y) * 0.5
    }));
    rank_sums_avg
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
/// Based on `scipy.stats.ranksums`.
fn rank_sums(x: &[i64], y: &[i64]) -> (f64, f64) {
    let n1 = x.len();
    let n2 = y.len();
    let all_data = [x, y].concat();
    let s: f64 = rank_data_avg(&all_data).into_iter().take(n1).sum();
    let n1 = n1 as f64;
    let n2 = n2 as f64;
    let expected = n1 * (n1 + n2 + 1.0) / 2.0;
    let z = (s - expected) / f64::sqrt(n1 * n2 * (n1 + n2 + 1.0) / 12.0);
    let prob = 2.0 * z.abs().norm_sf();
    (z, prob)
}

#[pyfunction]
#[pyo3(name = "rank_sums")]
pub fn rank_sums_py<'py>(
    _py: Python<'py>,
    x: PyReadonlyArray1<'_, i64>,
    y: PyReadonlyArray1<'_, i64>,
) -> (f64, f64) {
    let x = x.as_slice().expect("input not contiguous");
    let y = y.as_slice().expect("input not contiguous");

    let (z, p) = rank_sums(x, y);
    (z, p)
}

#[pyfunction]
#[pyo3(name = "rank_sums_2d")]
pub fn rank_sums_2d_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, i64>,
    y: PyReadonlyArray2<'_, i64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let mut b: Vec<(usize, f64, f64)> = x
        .as_array()
        .rows()
        .into_iter()
        .zip(y.as_array().rows().into_iter())
        .enumerate()
        .par_bridge()
        .map(|(i, (x, y))| {
            let (z, p) = rank_sums(x.as_slice().unwrap(), y.as_slice().unwrap());
            (i, z, p)
        })
        .collect::<Vec<(usize, f64, f64)>>();
    b.sort_unstable_by_key(|(i, _z, _p)| i.clone());
    let (z, p): (Vec<_>, Vec<_>) = b.into_iter().map(|(_i, z, p)| (z, p)).unzip();
    let z = Array1::from_vec(z);
    let p = Array1::from_vec(p);
    (z.into_pyarray(py), p.into_pyarray(py))
}
