use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use voracious_radix_sort::RadixSort;

#[pyfunction]
#[pyo3(name = "sort_1d_i8")]
/// Sort an 1D int8 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_i8_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i8>,
) -> Bound<'py, PyArray1<i8>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_i16")]
/// Sort an 1D int16 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_i16_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i16>,
) -> Bound<'py, PyArray1<i16>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_i32")]
/// Sort an 1D int32 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_i32_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i32>,
) -> Bound<'py, PyArray1<i32>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_i64")]
/// Sort an 1D int64 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_i64_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_u8")]
/// Sort an 1D uint8 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_u8_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u8>,
) -> Bound<'py, PyArray1<u8>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_u16")]
/// Sort an 1D uint16 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_u16_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u16>,
) -> Bound<'py, PyArray1<u16>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_u32")]
/// Sort an 1D uint32 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_u32_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u32>,
) -> Bound<'py, PyArray1<u32>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_u64")]
/// Sort an 1D uint64 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_u64_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u64>,
) -> Bound<'py, PyArray1<u64>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_f32")]
/// Sort an 1D float32 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_f32_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f32>,
) -> Bound<'py, PyArray1<f32>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "sort_1d_f64")]
/// Sort an 1D float64 array.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
pub fn sort_1d_f64_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let len = arr.len().unwrap();
    let mut arr_sorted = Vec::with_capacity(len);
    unsafe {
        arr_sorted.set_len(len);
    }
    arr_sorted.clone_from_slice(&arr.as_array().as_slice().unwrap());
    arr_sorted.voracious_sort();
    arr_sorted.into_pyarray(py)
}
