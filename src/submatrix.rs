use na::{ComplexField, Dyn};
use nalgebra as na;
use nalgebra::linalg::SVD as nalgebra_SVD;
use ndarray::iter::Iter;
/**
 * File: /src/Submatrix.rs
 * Created Date: Monday, January 22nd 2024
 * Author: Zihan
 * -----
 * Last Modified: Sunday, 28th January 2024 8:25:36 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

/// Extend the usage of slice for ndarray
/// struct Submatrix and impl
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{Lapack, SVDInplace, Scalar, SVD};
use ndarray_rand::rand_distr::num_traits::real::Real;
use std::ops::{Div, Index};
use nalgebra::DMatrix;

use crate::cocluster;

use crate::submatrix;

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
    T: ComplexField,
    <T as ComplexField>::RealField: Into<<T as Scalar>::Real>,
{
    pub data: ArrayView2<'a, T>,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub score: Option<<T as Scalar>::Real>,
}

impl<'a, T> Submatrix<'a, T>
where
    T: Scalar + Lapack,
    T: Div,
    T: ComplexField,
    <T as ComplexField>::RealField: Into<<T as Scalar>::Real>,
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

        let rm = Iterator::max(row_indices.iter())?;
        let cm = Iterator::max(col_indices.iter())?;

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

    // give `new` a nickname: from_indices to pub
    pub fn from_indices(
        matrix: &'a Array2<T>,
        row_indices: &[usize],
        col_indices: &[usize],
    ) -> Option<Self> {
        let row_indices = row_indices.to_vec();
        let col_indices = col_indices.to_vec();
        Self::new(matrix, row_indices, col_indices)
    }


    fn clone_to_dmatrix(&self) -> DMatrix<T>
    {
        let submatrix: Array2<T> = self
        .data
        .select(ndarray::Axis(0), &self.row_indices)
        .select(ndarray::Axis(1), &self.col_indices);

        let nrows = submatrix.view().nrows();
        let ncols = submatrix.view().ncols();
        let elements = submatrix.view().iter().cloned().collect::<Vec<T>>();
        DMatrix::from_vec(nrows, ncols, elements)
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

        // calculate svd and get the first two singular values

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
    T: ComplexField,
    <T as ComplexField>::RealField: Into<<T as Scalar>::Real>,
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
    T: ComplexField,
    <T as ComplexField>::RealField: Into<<T as Scalar>::Real>,
    Submatrix<'a, T>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}

// impl SVD in ndarray_linalg for Submatrix
impl<'a, T> SVD for Submatrix<'a, T>
where
    T: Scalar + Lapack,
    T: ComplexField,
    <T as ComplexField>::RealField: Into<<T as Scalar>::Real>,
{
    type U = Array2<T>;
    type VT = Array2<T>;
    type Sigma = Array1<<T as Scalar>::Real>;

    fn svd(
        &self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>), LinalgError> {
        // 提取子矩阵
        // self.data.
        //     select(ndarray::Axis(0), &self.row_indices)
        //     .select(ndarray::Axis(1), &self.col_indices)
        //     .to_owned()
        //     .svd_inplace(calc_u, calc_vt)

        // dbg!(&submatrix);

        // match submatrix.as_slice() {
        //     None => {
        //         return Err(LinalgError::MemoryNotCont)
        //     }
        //     Some(_) => (),
        // }

        let na_matrix = self.clone_to_dmatrix();
        let svd_result = na_matrix.svd(true, true);
        let u: na::Matrix<T, Dyn, Dyn, na::VecStorage<T, Dyn, Dyn>> = svd_result.u.unwrap(); // shaped as (row, row)
        let vt: na::Matrix<T, Dyn, Dyn, na::VecStorage<T, Dyn, Dyn>> = svd_result.v_t.unwrap(); // shaped as (col, col)
        let v: na::Matrix<T, Dyn, Dyn, na::VecStorage<T, Dyn, Dyn>> = vt.transpose(); // shaped as (col, row)

        let u_ndarray: Array2<T> =
            Array2::from_shape_vec((u.nrows(), u.ncols()), u.data.as_vec().clone()).unwrap();
        let v_ndarray: Array2<T> =
            Array2::from_shape_vec((v.nrows(), v.ncols()), v.data.as_vec().clone()).unwrap();

        let v_ndarray: Array2<T> = v_ndarray.t().to_owned();

        //type of svd_result.singular_values: OVector<T::RealField, DimMinimum<R, C>>
        // let s_ndarray: Array1<<T as Scalar>::Real> = Array1::from_shape_vec(
        //     (svd_result.singular_values.len()),
        //     svd_result.singular_values.data.as_vec().clone(),
        // ).unwrap();

        // add 0 image parts to s_ndarray
        let tmp: Array1<<T as ComplexField>::RealField> = Array1::from_shape_vec(
            (svd_result.singular_values.len()),
            svd_result.singular_values.data.as_vec().clone(),
        )
        .unwrap();

        let s_ndarray: Array1<<T as Scalar>::Real> = convert::<T>(tmp);

        Ok((Some(u_ndarray), s_ndarray, Some(v_ndarray)))
    }
}

fn convert<T>(array: Array1<<T as ComplexField>::RealField>) -> Array1<<T as Scalar>::Real>
where
    T: ComplexField + Scalar,
    <T as ComplexField>::RealField: Into<<T as Scalar>::Real>,
{
    array.into_iter().map(|x| x.into()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test for reslice of a submatrix to a new submatrix
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
        let a: Array2<f32> =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        let b = Submatrix::from_indices(&a, &[0, 2], &[1, 2]).unwrap();

        // println!("{:}", b);

        let (_, s, _) = b.svd(false, false).unwrap();

        // println!("{:?}", s);

        // roughly equal to 0.0
        assert!((s[0] - 12.5607).abs() < 1e-3);
    }
}
