use std::ops::{Div, Index};

/**
 * File: /src/Submatrix.rs
 * Created Date: Monday January 22nd 2024
 * Author: Zihan
 * -----
 * Last Modified: Sunday, 28th January 2024 6:48:07 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

/// Extend the usage of slice for ndarray
/// struct Submatrix and impl
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_linalg::{Lapack, SVDInplace, Scalar, SVD};
// use nalgebra::linalg::SVD;

///
/// # Example
/// ```
/// use fast_cocluster::submatrix::Submatrix;
/// use ndarray::Array2;
/// let a = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
/// let b = Submatrix::from_indices(&a, &[0, 2], &[1, 2]).unwrap();
///
/// assert_eq!(b[(1, 1)], 9);
/// assert_eq!(b[(0, 0)], 2);
/// ```
/// b = [[2, 3],
///     [8, 9]]
pub struct Submatrix<'a, T> 
where
    T: Scalar + Lapack,
{
    pub data:        ArrayView2<'a, T>,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub score:       Option<<T as Scalar>::Real>,
}

impl<'a, T> Submatrix<'a, T> 
where
    T: Scalar + Lapack,
    T: Div
{
    pub fn new(
        matrix: &'a Array2<T>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> Option<Self> {
        let data = matrix.view();

        // check if row_indices and col_indices are valid
        let row_max = matrix.shape()[0];
        let col_max = matrix.shape()[1];

        let rm = row_indices.iter().max().unwrap();
        let cm = col_indices.iter().max().unwrap();

        let score = None;

        if rm >= &row_max || cm >= &col_max {
            None
        } else {
            Some(Submatrix {
                data,
                row_indices,
                col_indices,
                score,
            })
        }
    }

    // give `new` a nick name: from_indices to pub
    pub fn from_indices(
        matrix: &'a Array2<T>,
        row_indices: &[usize],
        col_indices: &[usize],
    ) -> Option<Self> {
        let row_indices = row_indices.to_vec();
        let col_indices = col_indices.to_vec();
        Self::new(matrix, row_indices, col_indices)
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.row_indices.get(row).and_then(|&r| {
            self.col_indices
                .get(col)
                .and_then(|&c| self.data.get((r, c)))
        })
    }

    pub fn update_score(&mut self) {
        // if submatrix is smaller than 3*3, score gives an inf
        let row_len = self.row_indices.len();
        let col_len = self.col_indices.len();
        if row_len < 3 || col_len < 3 {
            self.score = None;
            return;
        }

        // calculate svd and get first two singular values

        let svd_result = self.svd(true, true).unwrap(); // Unwrap once and store
        let s1 = svd_result.1[0]; // Access the elements of the unwrapped result
        let s2 = svd_result.1[1];
        

        // calculate score
        self.score = Some(s1 / s2);
    }

    // fn slice(&self, new_row_indices: Vec<usize>, new_col_indices: Vec<usize>) -> Self {
    //     Submatrix::new(&self.data, new_row_indices, new_col_indices)
    // }
}

impl<'a, T> Index<(usize, usize)> for Submatrix<'a, T> 
where
    T: Scalar + Lapack,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1).unwrap()
    }
}

impl std::fmt::Debug for Submatrix<'_, f32> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let mut s = String::new();
        for i in 0..self.row_indices.len() {
            s.push_str("[");
            for j in 0..self.col_indices.len() {
                s.push_str(&format!("{}, ", &self[(i, j)]));
            }
            s.push_str("]\n");
        }

        write!(f, "{}", s)
    }
}


// impl Display
/// # Example
/// ```log
/// [2, 3]
/// [8, 9]
/// ```
impl<'a, T> std::fmt::Display for Submatrix<'a, T>
where
    T: std::fmt::Display,
    T: Scalar + Lapack,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.row_indices.len() {
            s.push_str("[");
            for j in 0..self.col_indices.len() {
                s.push_str(&format!("{}, ", &self[(i, j)]));
            }
            s.push_str("]\n");
        }

        write!(f, "{}", s)
    }
}

// impl SVD in ndarray_linalg for Submatrix
impl<'a, T> SVD for Submatrix<'a, T> 
where
    T: Scalar + Lapack,
{
    type U = Array2<T>;
    type Sigma = Array1<<T as Scalar>::Real>;
    type VT = Array2<T>;

    fn svd(
        &self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>), ndarray_linalg::error::LinalgError>
    {
        // 提取子矩阵
        self.data.
            select(ndarray::Axis(0), &self.row_indices)
            .select(ndarray::Axis(1), &self.col_indices)
            .to_owned()
            .svd_inplace(calc_u, calc_vt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// test for reslice of a submatrix to a new submatrix
    ///
    /// A = \[\[1, 2, 3\],
    ///     \[4, 5, 6\],
    ///    \[7, 8, 9\]\]
    ///
    /// B = A\[\[0, 2\], \[1, 2\]\]
    ///
    /// C = B\[\[1\], \[1\]\]
    ///
    /// C = \[\[9\]\]
    fn test_new_submatrix() {
        let a: Array2<f32> = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ).unwrap();

        let b = Submatrix::from_indices(&a, &[0, 2], &[1, 2]).unwrap();

        let (_, s, _) = b.svd(false, false).unwrap();

        // roughly equal to 0.0
        assert!((s[0] - 0.0).abs() < 1e-6);

        todo!()
    }
}
