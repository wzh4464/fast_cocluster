/**
 * File: /src/util.rs
 * Created Date: Tuesday, June 18th 2024
 * Author: Zihan
 * -----
 * Last Modified: Tuesday, 18th June 2024 11:12:25 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/
use std::collections::HashMap;
extern crate nalgebra as na;
use na::DMatrix;

/// 检查两个分类结果是否等价
pub fn are_equivalent_classifications(a: Vec<usize>, b: Vec<usize>) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut a_to_b_map = HashMap::new();
    let mut b_to_a_map = HashMap::new();

    for (&a_class, &b_class) in a.iter().zip(b.iter()) {
        let a_mapped = a_to_b_map.entry(a_class).or_insert(b_class);
        let b_mapped = b_to_a_map.entry(b_class).or_insert(a_class);

        if a_mapped != &b_class || b_mapped != &a_class {
            return false;
        }
    }

    true
}

pub fn clone_to_dmatrix<T>(array_view: ndarray::ArrayView2<T>) -> DMatrix<T>
where
    T: Clone,
    T: na::Scalar,
{
    let nrows = array_view.ncols();
    let ncols = array_view.nrows();
    let elements = array_view.iter().cloned().collect::<Vec<T>>();
    DMatrix::from_vec(nrows, ncols, elements).transpose()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_are_equivalent_classifications() {
        assert!(are_equivalent_classifications(
            vec![0, 2, 1, 1],
            vec![1, 2, 0, 0]
        ));
        assert!(are_equivalent_classifications(
            vec![0, 1, 1, 2],
            vec![1, 2, 2, 0]
        ));
        assert!(!are_equivalent_classifications(
            vec![0, 1, 1, 2],
            vec![1, 2, 0, 0]
        ));
        assert!(!are_equivalent_classifications(
            vec![0, 1, 1],
            vec![1, 2, 0, 0]
        ));
    }
    /// Test for cloning an ndarray array view into a nalgebra DMatrix.
    #[test]
    fn test_clone_to_dmatrix() {
        // 创建一个ndarray矩阵
        let array = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // 获取矩阵的视图
        let array_view = array.view();

        // 使用clone_to_dmatrix函数将视图转换为DMatrix
        let dmatrix = clone_to_dmatrix(array_view);

        // 检查形状是否一致
        assert_eq!(array_view.nrows(), dmatrix.nrows());
        assert_eq!(array_view.ncols(), dmatrix.ncols());

        // 检查每个元素是否相等
        for i in 0..array_view.nrows() {
            for j in 0..array_view.ncols() {
                assert_eq!(array_view[(i, j)], dmatrix[(i, j)]);
            }
        }
    }
}
