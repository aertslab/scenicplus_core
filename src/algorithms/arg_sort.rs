use std::cmp::Ordering;
use voracious_radix_sort::{RadixSort, Radixable};

// We need a struct.
// We want, for example, to sort these structs by the key: "value".
// This struct must implement the Copy and Clone traits, we can just derive them.
// For the multithread version the struct must implement de `Send` and `Sync` traits
// too, which are by default for primitive types.
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
        match self.len() >= 2000 {
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
        match self.len() >= 2000 {
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
        match self.len() >= 2000 {
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
        // For some reason radix sort is only faster from 9000 elements for i64
        // and not from 2000 elements like for other primitives.
        match self.len() >= 9000 {
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
        match self.len() >= 2000 {
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
        match self.len() >= 2000 {
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
        match self.len() >= 2000 {
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
        match self.len() >= 2000 {
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
        match self.len() >= 2000 {
            false => self.arg_sort(),
            true => self.arg_sort_radix(),
        }
    }
}
