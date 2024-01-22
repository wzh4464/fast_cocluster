/**
 * File: /src/Submatrix.rs
 * Created Date: Monday January 22nd 2024
 * Author: Zihan
 * -----
 * Last Modified: Monday, 22nd January 2024 4:46:21 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/

/// Extend the usage of slice for ndarray
/// struct Submatrix and impl
use ndarray::{Array2, ArrayView2};
use std::ops::Index;

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
pub struct Submatrix<'a, T> {
    data: ArrayView2<'a, T>,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
}

impl<'a, T> Submatrix<'a, T> {
    fn new(
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

        if rm >= &row_max || cm >= &col_max {
            None
        } else {
            Some(Submatrix {
                data,
                row_indices,
                col_indices,
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

    // fn slice(&self, new_row_indices: Vec<usize>, new_col_indices: Vec<usize>) -> Self {
    //     Submatrix::new(&self.data, new_row_indices, new_col_indices)
    // }
}

impl<'a, T> Index<(usize, usize)> for Submatrix<'a, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1).unwrap()
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
        let a = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let b = Submatrix::new(&a, vec![0, 2], vec![1, 2]).unwrap();
        assert_eq!(b[(1, 1)], 9);
    }
}
