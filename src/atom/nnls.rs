use ndarray::{s, Array1, Array2, Axis};

/// Result from NNLS solver
pub struct NnlsResult {
    pub x: Array2<f64>,
    pub success: bool,
}

/// Nonnegativity-constrained least squares with block principal pivoting.
///
/// Solves min ||A*X - B||_2^2 s.t. X >= 0 element-wise.
///
/// Port of Kim & Park 2011 (SISC vol.33 no.6 pp.3261-3281).
///
/// A: (m, n), B: (m, k) → X: (n, k)
pub fn nnlsm_blockpivot(
    a: &Array2<f64>,
    b: &Array2<f64>,
    init: Option<&Array2<f64>>,
) -> NnlsResult {
    let ata = a.t().dot(a);
    let atb = a.t().dot(b);

    nnlsm_blockpivot_product(&ata, &atb, init)
}

/// Block-pivoting NNLS on pre-computed A^T*A and A^T*B.
fn nnlsm_blockpivot_product(
    ata: &Array2<f64>,
    atb: &Array2<f64>,
    init: Option<&Array2<f64>>,
) -> NnlsResult {
    let n = atb.nrows();
    let k = atb.ncols();
    let max_iter = n * 5;

    let mut x: Array2<f64>;
    let mut y: Array2<f64>;
    let mut pass_set: Array2<bool>;

    if let Some(init_val) = init {
        pass_set = init_val.mapv(|v| v > 0.0);
        x = normal_eq_comb(ata, atb, Some(&pass_set));
        y = ata.dot(&x) - atb;
    } else {
        x = Array2::zeros((n, k));
        y = -atb.clone();
        pass_set = Array2::from_elem((n, k), false);
    }

    let p_bar: i32 = 3;
    let mut p_vec = Array1::from_elem(k, p_bar);
    let mut ninf_vec = Array1::from_elem(k, (n + 1) as i32);

    // not_opt_set: Y < 0 and not in PassSet
    let not_opt_set = compute_not_opt_set(&y, &pass_set);
    // infea_set: X < 0 and in PassSet
    let infea_set = compute_infea_set(&x, &pass_set);

    let mut not_good = sum_columns_bool(&not_opt_set) + &sum_columns_bool(&infea_set);
    let mut not_opt_colset: Array1<bool> = not_good.mapv(|v| v > 0);
    let mut not_opt_cols: Vec<usize> = not_opt_colset
        .iter()
        .enumerate()
        .filter(|(_, &v)| v)
        .map(|(i, _)| i)
        .collect();

    let mut big_iter = 0;
    let mut success = true;

    while !not_opt_cols.is_empty() {
        big_iter += 1;
        if max_iter > 0 && big_iter > max_iter {
            success = false;
            break;
        }

        // Classify columns
        let mut cols1 = Vec::new();
        let mut cols2 = Vec::new();
        let mut cols3 = Vec::new();

        for &col in &not_opt_cols {
            if not_good[col] < ninf_vec[col] {
                cols1.push(col);
            } else if p_vec[col] >= 1 {
                cols2.push(col);
            } else {
                cols3.push(col);
            }
        }

        // Process cols1: reset p_bar, update ninf
        for &col in &cols1 {
            p_vec[col] = p_bar;
            ninf_vec[col] = not_good[col];
            for row in 0..n {
                if y[[row, col]] < 0.0 && !pass_set[[row, col]] {
                    pass_set[[row, col]] = true;
                }
                if x[[row, col]] < 0.0 && pass_set[[row, col]] {
                    pass_set[[row, col]] = false;
                }
            }
        }

        // Process cols2: decrement p_vec
        for &col in &cols2 {
            p_vec[col] -= 1;
            for row in 0..n {
                if y[[row, col]] < 0.0 && !pass_set[[row, col]] {
                    pass_set[[row, col]] = true;
                }
                if x[[row, col]] < 0.0 && pass_set[[row, col]] {
                    pass_set[[row, col]] = false;
                }
            }
        }

        // Process cols3: backup rule — flip one entry
        for &col in &cols3 {
            let mut to_change = None;
            for row in (0..n).rev() {
                if (y[[row, col]] < 0.0 && !pass_set[[row, col]])
                    || (x[[row, col]] < 0.0 && pass_set[[row, col]])
                {
                    to_change = Some(row);
                    break;
                }
            }
            if let Some(row) = to_change {
                pass_set[[row, col]] = !pass_set[[row, col]];
            }
        }

        // Solve normal equations for not_opt columns
        let sub_atb = select_columns(atb, &not_opt_cols);
        let sub_pass = select_columns_bool(&pass_set, &not_opt_cols);
        let sub_x = normal_eq_comb(ata, &sub_atb, Some(&sub_pass));

        // Write back
        for (j, &col) in not_opt_cols.iter().enumerate() {
            for row in 0..n {
                x[[row, col]] = if sub_x[[row, j]].abs() < 1e-12 {
                    0.0
                } else {
                    sub_x[[row, j]]
                };
            }
        }

        // Update Y for not_opt columns
        let sub_x_updated = select_columns(&x, &not_opt_cols);
        let sub_y = ata.dot(&sub_x_updated) - &sub_atb;
        for (j, &col) in not_opt_cols.iter().enumerate() {
            for row in 0..n {
                y[[row, col]] = if sub_y[[row, j]].abs() < 1e-12 {
                    0.0
                } else {
                    sub_y[[row, j]]
                };
            }
        }

        // Recompute not_opt and infeasible sets
        let not_opt_set = compute_not_opt_set(&y, &pass_set);
        let infea_set = compute_infea_set(&x, &pass_set);
        not_good = sum_columns_bool(&not_opt_set) + &sum_columns_bool(&infea_set);

        // Only recheck columns that were not_opt
        for col in 0..k {
            if !not_opt_colset[col] {
                not_good[col] = 0;
            }
        }

        not_opt_colset = not_good.mapv(|v| v > 0);
        not_opt_cols = not_opt_colset
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i)
            .collect();
    }

    NnlsResult { x, success }
}

/// Solve normal equations with combinatorial grouping.
/// For columns with matching PassSet patterns, solve once and share the solution.
fn normal_eq_comb(
    ata: &Array2<f64>,
    atb: &Array2<f64>,
    pass_set: Option<&Array2<bool>>,
) -> Array2<f64> {
    let n = atb.nrows();
    let k = atb.ncols();

    match pass_set {
        None => {
            // Solve AtA * X = AtB directly
            solve_linear(ata, atb)
        }
        Some(ps) => {
            if ps.iter().all(|&v| v) {
                return solve_linear(ata, atb);
            }

            let mut z = Array2::zeros((n, k));

            // Group columns by their PassSet pattern
            let groups = column_group(ps);

            for group in groups {
                if group.is_empty() {
                    continue;
                }
                let col0 = group[0];
                let cols: Vec<usize> = (0..n).filter(|&r| ps[[r, col0]]).collect();
                if cols.is_empty() {
                    continue;
                }

                // Extract sub-system
                let sub_ata = extract_submatrix(ata, &cols, &cols);
                let sub_atb = extract_submatrix_cols(atb, &cols, &group);

                let sub_z = solve_linear(&sub_ata, &sub_atb);

                // Write back
                for (gi, &gc) in group.iter().enumerate() {
                    for (ri, &rc) in cols.iter().enumerate() {
                        z[[rc, gc]] = sub_z[[ri, gi]];
                    }
                }
            }

            z
        }
    }
}

/// Group columns of a boolean matrix by identical patterns
fn column_group(b: &Array2<bool>) -> Vec<Vec<usize>> {
    let k = b.ncols();
    if k == 0 {
        return vec![];
    }

    let mut groups: Vec<Vec<usize>> = vec![vec![0]];
    let mut patterns: Vec<Vec<bool>> = vec![b.column(0).to_vec()];

    for col in 1..k {
        let pat: Vec<bool> = b.column(col).to_vec();
        let mut found = false;
        for (gi, existing) in patterns.iter().enumerate() {
            if *existing == pat {
                groups[gi].push(col);
                found = true;
                break;
            }
        }
        if !found {
            groups.push(vec![col]);
            patterns.push(pat);
        }
    }

    groups
}

/// Solve A*X = B using LU decomposition (via manual Gaussian elimination)
fn solve_linear(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let k = b.ncols();

    if n == 0 || k == 0 {
        return Array2::zeros((n, k));
    }

    // Augmented matrix [A | B]
    let mut aug = Array2::zeros((n, n + k));
    aug.slice_mut(s![.., ..n]).assign(a);
    aug.slice_mut(s![.., n..]).assign(b);

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-30 {
            continue; // Singular — skip
        }

        // Swap rows
        if max_row != col {
            for j in 0..(n + k) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..(n + k) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array2::zeros((n, k));
    for col in (0..n).rev() {
        let pivot = aug[[col, col]];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for j in 0..k {
            let mut sum = aug[[col, n + j]];
            for row in (col + 1)..n {
                sum -= aug[[col, row]] * x[[row, j]];
            }
            x[[col, j]] = sum / pivot;
        }
    }

    x
}

// Helper functions

fn compute_not_opt_set(y: &Array2<f64>, pass_set: &Array2<bool>) -> Array2<bool> {
    let mut result = Array2::from_elem(y.dim(), false);
    ndarray::Zip::from(&mut result)
        .and(y)
        .and(pass_set)
        .for_each(|r, &yv, &ps| {
            *r = yv < 0.0 && !ps;
        });
    result
}

fn compute_infea_set(x: &Array2<f64>, pass_set: &Array2<bool>) -> Array2<bool> {
    let mut result = Array2::from_elem(x.dim(), false);
    ndarray::Zip::from(&mut result)
        .and(x)
        .and(pass_set)
        .for_each(|r, &xv, &ps| {
            *r = xv < 0.0 && ps;
        });
    result
}

fn sum_columns_bool(a: &Array2<bool>) -> Array1<i32> {
    let k = a.ncols();
    let mut result = Array1::zeros(k);
    for col in 0..k {
        result[col] = a.column(col).iter().filter(|&&v| v).count() as i32;
    }
    result
}

fn select_columns(a: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let n = a.nrows();
    let mut result = Array2::zeros((n, cols.len()));
    for (j, &col) in cols.iter().enumerate() {
        result.column_mut(j).assign(&a.column(col));
    }
    result
}

fn select_columns_bool(a: &Array2<bool>, cols: &[usize]) -> Array2<bool> {
    let n = a.nrows();
    let mut result = Array2::from_elem((n, cols.len()), false);
    for (j, &col) in cols.iter().enumerate() {
        for row in 0..n {
            result[[row, j]] = a[[row, col]];
        }
    }
    result
}

fn extract_submatrix(a: &Array2<f64>, rows: &[usize], cols: &[usize]) -> Array2<f64> {
    let mut result = Array2::zeros((rows.len(), cols.len()));
    for (i, &r) in rows.iter().enumerate() {
        for (j, &c) in cols.iter().enumerate() {
            result[[i, j]] = a[[r, c]];
        }
    }
    result
}

fn extract_submatrix_cols(a: &Array2<f64>, rows: &[usize], cols: &[usize]) -> Array2<f64> {
    let mut result = Array2::zeros((rows.len(), cols.len()));
    for (i, &r) in rows.iter().enumerate() {
        for (j, &c) in cols.iter().enumerate() {
            result[[i, j]] = a[[r, c]];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_nnlsm_blockpivot_basic() {
        // A * X_org = B where X_org >= 0
        let m = 20;
        let n = 5;
        let k = 10;

        let a = Array2::random((m, n), Uniform::new(0.0, 1.0));
        let mut x_org = Array2::random((n, k), Uniform::new(0.0, 1.0));
        // Zero out some entries
        for i in 0..n {
            for j in 0..k {
                if x_org[[i, j]] < 0.3 {
                    x_org[[i, j]] = 0.0;
                }
            }
        }
        let b = a.dot(&x_org);

        let result = nnlsm_blockpivot(&a, &b, None);
        assert!(result.success);

        // Check solution is nonnegative
        for &v in result.x.iter() {
            assert!(v >= -1e-10, "Solution should be nonnegative, got {}", v);
        }

        // Check residual is small
        let residual = &a.dot(&result.x) - &b;
        let norm = residual.mapv(|v| v * v).sum().sqrt();
        let b_norm = b.mapv(|v| v * v).sum().sqrt();
        assert!(
            norm / b_norm < 1e-4,
            "Relative residual too large: {}",
            norm / b_norm
        );
    }
}
