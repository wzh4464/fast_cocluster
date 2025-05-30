/**
 * File: /src/matrix.rs
 * Created Date: Thursday, November 23rd 2023
 * Author: Zihan
 * -----
 * Last Modified: Monday, 26th May 2025 11:31:28 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

use std::ops::{Index, IndexMut};

// Array2
use ndarray::Array2;
use ndarray_rand::rand_distr::num_traits::Zero;

/// A generic matrix structure using `ndarray::Array2`.
pub struct Matrix<T> {
    pub data: Array2<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Matrix<T> {
    /// Creates a new `Matrix` from an `Array2`.
    ///
    /// # Arguments
    ///
    /// * `data` - An `Array2` containing the matrix data.
    ///
    /// # Returns
    ///
    /// * A new `Matrix` containing the provided data.
    fn new(data: Array2<T>) -> Matrix<T> {
        let rows = data.shape()[0];
        let cols = data.shape()[1];
        Matrix { data, rows, cols }
    }

    /// Returns a reference to the element at the specified row and column, or `None` if the indices are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index.
    /// * `col` - The column index.
    ///
    /// # Returns
    ///
    /// * An `Option` containing a reference to the element if it exists, or `None`.
    fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols {
            None
        } else {
            Some(&self.data[(row, col)])
        }
    }

    /// Returns a mutable reference to the element at the specified row and column, or `None` if the indices are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index.
    /// * `col` - The column index.
    ///
    /// # Returns
    ///
    /// * An `Option` containing a mutable reference to the element if it exists, or `None`.
    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.rows || col >= self.cols {
            None
        } else {
            Some(&mut self.data[(row, col)])
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.data[(row, col)]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.data[(row, col)]
    }
}

impl<T> Matrix<T>
where
    T: Clone + Zero,
{
    /// Creates a new `Matrix` containing a clone of the elements in the specified row and column ranges.
    ///
    /// # Arguments
    ///
    /// * `row_range` - A range specifying the rows to include in the new matrix.
    /// * `col_range` - A range specifying the columns to include in the new matrix.
    ///
    /// # Returns
    ///
    /// * A new `Matrix` containing the elements in the specified ranges.
    pub fn slice_clone(
        &self,
        row_range: &std::ops::Range<usize>,
        col_range: &std::ops::Range<usize>,
    ) -> Matrix<T> {
        let mut data = Array2::<T>::zeros((
            row_range.end - row_range.start,
            col_range.end - col_range.start,
        ));
        let mut vect: Vec<T> = Vec::new();

        for i in row_range.clone() {
            for j in col_range.clone() {
                vect.push(self[(i, j)].clone());
            }
        }

        data.assign(
            &Array2::from_shape_vec(
                (
                    row_range.end - row_range.start,
                    col_range.end - col_range.start,
                ),
                vect,
            )
            .unwrap(),
        );
        Matrix {
            data,
            rows: row_range.end - row_range.start,
            cols: col_range.end - col_range.start,
        }
    }
}

/// A structure representing a slice of a `Matrix`.
struct MatrixSlice<'a, T> {
    original_matrix: &'a mut Matrix<T>,
    row_range:       std::ops::Range<usize>,
    col_range:       std::ops::Range<usize>,
}

impl<'a, T> MatrixSlice<'a, T> {
    /// Creates a new `MatrixSlice`.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A mutable reference to the original matrix.
    /// * `row_range` - A range specifying the rows to include in the slice.
    /// * `col_range` - A range specifying the columns to include in the slice.
    ///
    /// # Returns
    ///
    /// * A new `MatrixSlice` containing references to the specified elements in the original matrix.
    pub fn new(
        matrix: &'a mut Matrix<T>,
        row_range: std::ops::Range<usize>,
        col_range: std::ops::Range<usize>,
    ) -> MatrixSlice<'a, T> {
        MatrixSlice {
            original_matrix: matrix,
            row_range,
            col_range,
        }
    }

    /// Returns a reference to the element at the specified row and column within the slice, or `None` if the indices are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index within the slice.
    /// * `col` - The column index within the slice.
    ///
    /// # Returns
    ///
    /// * An `Option` containing a reference to the element if it exists, or `None`.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.row_range.end - self.row_range.start
            || col >= self.col_range.end - self.col_range.start
        {
            None
        } else {
            Some(&self.original_matrix[(row + self.row_range.start, col + self.col_range.start)])
        }
    }

    /// Sets the value of the element at the specified row and column within the slice.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index within the slice.
    /// * `col` - The column index within the slice.
    /// * `value` - The value to set.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.original_matrix[(row + self.row_range.start, col + self.col_range.start)] = value;
    }
}

impl<'a, T> Index<(usize, usize)> for MatrixSlice<'a, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.original_matrix[(row + self.row_range.start, col + self.col_range.start)]
    }
}

impl<'a, T> IndexMut<(usize, usize)> for MatrixSlice<'a, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.original_matrix[(row + self.row_range.start, col + self.col_range.start)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_matrix() {
        let data = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let matrix = Matrix::new(data);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
    }

    #[test]
    fn test_get() {
        let data = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let matrix = Matrix::new(data);
        assert_eq!(matrix.get(0, 0), Some(&1));
        assert_eq!(matrix.get(0, 1), Some(&2));
        assert_eq!(matrix.get(1, 0), Some(&3));
        assert_eq!(matrix.get(1, 1), Some(&4));
        assert_eq!(matrix.get(2, 0), None);
        assert_eq!(matrix.get(0, 2), None);
    }

    #[test]
    fn test_get_mut() {
        let mut data = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let mut matrix = Matrix::new(data);
        assert_eq!(matrix.get_mut(0, 0), Some(&mut 1));
        assert_eq!(matrix.get_mut(0, 1), Some(&mut 2));
        assert_eq!(matrix.get_mut(1, 0), Some(&mut 3));
        assert_eq!(matrix.get_mut(1, 1), Some(&mut 4));
        assert_eq!(matrix.get_mut(2, 0), None);
        assert_eq!(matrix.get_mut(0, 2), None);
    }

    #[test]
    fn test_index() {
        let data = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let matrix = Matrix::new(data);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
    }

    #[test]
    fn test_index_mut() {
        let mut data = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let mut matrix = Matrix::new(data);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
        matrix[(0, 0)] = 5;
        matrix[(0, 1)] = 6;
        matrix[(1, 0)] = 7;
        matrix[(1, 1)] = 8;
        assert_eq!(matrix[(0, 0)], 5);
        assert_eq!(matrix[(0, 1)], 6);
        assert_eq!(matrix[(1, 0)], 7);
        assert_eq!(matrix[(1, 1)], 8);
    }

    #[test]
    fn test_slice() {
        // 3*3, select 1..3, 1..3
        let data = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let mut matrix = Matrix::new(data);
        let slice = MatrixSlice::new(&mut matrix, 1..3, 1..3);
        assert_eq!(slice.get(0, 0), Some(&5));
        assert_eq!(slice.get(0, 1), Some(&6));
        assert_eq!(slice.get(1, 0), Some(&8));
        assert_eq!(slice.get(1, 1), Some(&9));

        assert_eq!(slice[(0, 0)], 5);
        assert_eq!(slice[(0, 1)], 6);
        assert_eq!(slice[(1, 0)], 8);
        assert_eq!(slice[(1, 1)], 9);
    }

    #[test]
    fn test_slice_set() {
        // 3*3, select 1..3, 1..3
        let data = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let mut matrix = Matrix::new(data);
        let mut slice = MatrixSlice::new(&mut matrix, 1..3, 1..3);
        slice.set(0, 0, 10);
        slice.set(0, 1, 11);
        slice.set(1, 0, 12);
        slice.set(1, 1, 13);
        assert_eq!(slice.get(0, 0), Some(&10));
        assert_eq!(slice.get(0, 1), Some(&11));
        assert_eq!(slice.get(1, 0), Some(&12));
        assert_eq!(slice.get(1, 1), Some(&13));

        assert_eq!(slice[(0, 0)], 10);
        assert_eq!(slice[(0, 1)], 11);
        assert_eq!(slice[(1, 0)], 12);
        assert_eq!(slice[(1, 1)], 13);
    }

    #[test]
    fn test_slice_clone() {
        // 3*3, select 1..3, 1..3
        let data = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let mut matrix = Matrix::new(data);
        let slice = matrix.slice_clone(&{ 1..3 }, &{ 1..3 });
        assert_eq!(slice.get(0, 0), Some(&5));
        assert_eq!(slice.get(0, 1), Some(&6));
        assert_eq!(slice.get(1, 0), Some(&8));
        assert_eq!(slice.get(1, 1), Some(&9));

        assert_eq!(slice[(0, 0)], 5);
        assert_eq!(slice[(0, 1)], 6);
        assert_eq!(slice[(1, 0)], 8);
        assert_eq!(slice[(1, 1)], 9);
    }
}
