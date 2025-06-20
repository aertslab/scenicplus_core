use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::fmt;
use voracious_radix_sort::{RadixSort, Radixable};

// Define a struct to store indexed values for sorting operations.
// This struct enables sorting based on a specified key, such as the `value` field.
//
// The struct derives the `Copy` and `Clone` traits to allow efficient duplication and copying.
// Additionally, for use in a multithreaded context, it must implement the `Send` and `Sync` traits,
// which are automatically satisfied for primitive types.
#[derive(Copy, Clone, Debug)]
pub struct ArgSortRadix<T> {
    idx: usize,
    value: T,
}

impl<T: PartialOrd> PartialOrd for ArgSortRadix<T> {
    fn partial_cmp(&self, other: &ArgSortRadix<T>) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: PartialEq> PartialEq for ArgSortRadix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Radixable<i8> for ArgSortRadix<i8> {
    type Key = i8;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<i16> for ArgSortRadix<i16> {
    type Key = i16;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<i32> for ArgSortRadix<i32> {
    type Key = i32;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<i64> for ArgSortRadix<i64> {
    type Key = i64;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<u8> for ArgSortRadix<u8> {
    type Key = u8;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<u16> for ArgSortRadix<u16> {
    type Key = u16;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<u32> for ArgSortRadix<u32> {
    type Key = u32;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<u64> for ArgSortRadix<u64> {
    type Key = u64;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<f32> for ArgSortRadix<f32> {
    type Key = f32;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

impl Radixable<f64> for ArgSortRadix<f64> {
    type Key = f64;
    #[inline]
    fn key(&self) -> Self::Key {
        self.value
    }
}

pub trait ArgSort<T: PartialOrd> {
    fn arg_sort(&self) -> Vec<usize>;
    fn arg_sort_radix(&self) -> Vec<usize>;
    fn arg_sort_fastest(&self) -> Vec<usize>;
}

impl ArgSort<i8> for &[i8] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by_key(|&i| unsafe { self.get_unchecked(i) });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 400 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<i16> for &[i16] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by_key(|&i| unsafe { self.get_unchecked(i) });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 1300 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<i32> for &[i32] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by_key(|&i| unsafe { self.get_unchecked(i) });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 2500 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<i64> for &[i64] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by_key(|&i| unsafe { self.get_unchecked(i) });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 10000 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<u8> for &[u8] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by_key(|&i| unsafe { self.get_unchecked(i) });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 400 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<u16> for &[u16] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by_key(|&i| unsafe { self.get_unchecked(i) });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 1200 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<u32> for &[u32] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by_key(|&i| unsafe { self.get_unchecked(i) });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 1200 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<u64> for &[u64] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by_key(|&i| unsafe { self.get_unchecked(i) });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 1200 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<f32> for &[f32] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&i1, &i2| {
            unsafe { self.get_unchecked(i1) }
                .partial_cmp(unsafe { self.get_unchecked(i2) })
                .unwrap()
        });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 2000 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

impl ArgSort<f64> for &[f64] {
    fn arg_sort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&i1, &i2| {
            unsafe { self.get_unchecked(i1) }
                .partial_cmp(unsafe { self.get_unchecked(i2) })
                .unwrap()
        });
        indices
    }

    fn arg_sort_radix(&self) -> Vec<usize> {
        let mut idx_value_data = self
            .iter()
            .enumerate()
            .map(|(i, v)| ArgSortRadix { idx: i, value: *v })
            .collect::<Vec<_>>();
        idx_value_data.voracious_sort();
        idx_value_data.into_iter().map(|x| x.idx).collect()
    }

    fn arg_sort_fastest(&self) -> Vec<usize> {
        match self.len() >= 5000 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
enum ArgSortMethod {
    Standard,
    Radix,
    Fastest,
}

impl fmt::Display for ArgSortMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ArgSortMethod::Standard => write!(f, "standard"),
            ArgSortMethod::Radix => write!(f, "radix"),
            ArgSortMethod::Fastest => write!(f, "fastest"),
        }
    }
}

impl TryFrom<&str> for ArgSortMethod {
    type Error = ();

    fn try_from(method: &str) -> Result<Self, Self::Error> {
        match method {
            "standard" => Ok(ArgSortMethod::Standard),
            "radix" => Ok(ArgSortMethod::Radix),
            "fastest" => Ok(ArgSortMethod::Fastest),
            _ => Err(()),
        }
    }
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_i8")]
/// Returns the indices that would sort an 1D int8 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_i8_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i8>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_i16")]
/// Returns the indices that would sort an 1D int16 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_i16_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i16>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_i32")]
/// Returns the indices that would sort an 1D int32 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_i32_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i32>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_i64")]
/// Returns the indices that would sort an 1D int64 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_i64_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_u8")]
/// Returns the indices that would sort an 1D uint8 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_u8_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u8>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_u16")]
/// Returns the indices that would sort an 1D uint16 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_u16_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u16>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_u32")]
/// Returns the indices that would sort an 1D uint32 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_u32_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u32>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_u64")]
/// Returns the indices that would sort an 1D uint64 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_u64_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u64>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_f32")]
/// Returns the indices that would sort an 1D float32 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_f32_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f32>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "arg_sort_1d_f64")]
/// Returns the indices that would sort an 1D float64 array.
///
/// Perform an indirect sort using the algorithm specified by the `method` keyword.
/// It returns an array of indices of the same shape as `arr` that index data in
/// sorted order.
///
/// Parameters
/// ----------
/// arr
///     1D continuous numpy array.
/// reverse
///     If `True`, sort in descending order.
/// method
///     Method to use for sorting: "standard", "radix", "fastest"
///     (uses standard or radix sort depending on the size).
///
pub fn arg_sort_1d_f64_py<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    reverse: bool,
    method: &str,
) -> Bound<'py, PyArray1<usize>> {
    let method = ArgSortMethod::try_from(method).unwrap_or(ArgSortMethod::Fastest);

    let mut arg_sorted = match method {
        ArgSortMethod::Standard => arr.as_array().as_slice().unwrap().arg_sort(),
        ArgSortMethod::Radix => arr.as_array().as_slice().unwrap().arg_sort_radix(),
        ArgSortMethod::Fastest => arr.as_array().as_slice().unwrap().arg_sort_fastest(),
    };

    if reverse {
        arg_sorted.reverse();
    };

    arg_sorted.into_pyarray(py)
}
