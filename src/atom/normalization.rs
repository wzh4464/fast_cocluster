use ndarray::{Array1, Array2, Axis};

use super::update_rules::nan_to_num;

/// Standard normalization (used by NBVD, ONM3F, ONMTF):
///   DF = diag(F.sum(axis=0) ^ -0.5)
///   F  = F * DF
///   S  = F^T * X * G
///   DG = diag(G.sum(axis=0) ^ -0.5)
///   G  = G * DG
pub fn normalize_standard(
    x: &Array2<f64>,
    f: &mut Array2<f64>,
    s: &mut Array2<f64>,
    g: &mut Array2<f64>,
) {
    // DF = diag(col_sums^{-0.5})
    let f_col_sums = f.sum_axis(Axis(0));
    let df: Array1<f64> = f_col_sums.mapv(|v| if v.abs() < 1e-30 { 0.0 } else { v.powf(-0.5) });
    // F = F * DF (scale each column)
    for (j, &scale) in df.iter().enumerate() {
        f.column_mut(j).mapv_inplace(|v| v * scale);
    }

    // S = F^T * X * G
    *s = f.t().dot(x).dot(&*g);

    // DG = diag(col_sums^{-0.5})
    let g_col_sums = g.sum_axis(Axis(0));
    let dg: Array1<f64> = g_col_sums.mapv(|v| if v.abs() < 1e-30 { 0.0 } else { v.powf(-0.5) });
    for (j, &scale) in dg.iter().enumerate() {
        g.column_mut(j).mapv_inplace(|v| v * scale);
    }

    nan_to_num(f);
    nan_to_num(s);
    nan_to_num(g);
}

/// PNMTF normalization variant:
///   DF = diag(F.sum(axis=0))
///   DG = diag(G.sum(axis=0))
///   F  = F * diag(F.sum(axis=0)^{-1})
///   S  = DF * S * DG
///   G  = G * diag(G.sum(axis=0)^{-1})
pub fn normalize_pnmtf(f: &mut Array2<f64>, s: &mut Array2<f64>, g: &mut Array2<f64>) {
    let f_col_sums = f.sum_axis(Axis(0));
    let g_col_sums = g.sum_axis(Axis(0));

    // S = DF * S * DG
    // DF is diag(f_col_sums), DG is diag(g_col_sums)
    // (DF * S)_{ij} = f_col_sums[i] * S_{ij}
    // (DF * S * DG)_{ij} = f_col_sums[i] * S_{ij} * g_col_sums[j]
    let rows = s.nrows();
    let cols = s.ncols();
    for i in 0..rows {
        for j in 0..cols {
            s[[i, j]] *= f_col_sums[i] * g_col_sums[j];
        }
    }

    // F = F * diag(f_col_sums^{-1})
    for (j, &cs) in f_col_sums.iter().enumerate() {
        let inv = if cs.abs() < 1e-30 { 0.0 } else { 1.0 / cs };
        f.column_mut(j).mapv_inplace(|v| v * inv);
    }

    // G = G * diag(g_col_sums^{-1})
    for (j, &cs) in g_col_sums.iter().enumerate() {
        let inv = if cs.abs() < 1e-30 { 0.0 } else { 1.0 / cs };
        g.column_mut(j).mapv_inplace(|v| v * inv);
    }

    nan_to_num(f);
    nan_to_num(s);
    nan_to_num(g);
}
