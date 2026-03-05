use ndarray::Array2;

/// Elementwise multiplicative update: base * numer / (denom + eps)
pub fn multiplicative_update(
    base: &Array2<f64>,
    numer: &Array2<f64>,
    denom: &Array2<f64>,
    eps: f64,
) -> Array2<f64> {
    let mut result = base.clone();
    ndarray::Zip::from(&mut result)
        .and(numer)
        .and(denom)
        .for_each(|r, &n, &d| {
            *r *= n / (d + eps);
        });
    nan_to_num(&mut result);
    result
}

/// Elementwise sqrt multiplicative update: base * (numer / (denom + eps))^0.5
pub fn sqrt_multiplicative_update(
    base: &Array2<f64>,
    numer: &Array2<f64>,
    denom: &Array2<f64>,
    eps: f64,
) -> Array2<f64> {
    let mut result = base.clone();
    ndarray::Zip::from(&mut result)
        .and(numer)
        .and(denom)
        .for_each(|r, &n, &d| {
            *r *= (n / (d + eps)).sqrt();
        });
    nan_to_num(&mut result);
    result
}

/// Replace NaN and Inf with 0.0 in-place
pub fn nan_to_num(a: &mut Array2<f64>) {
    a.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
}

/// Frobenius norm squared: ||X - F*S*G^T||_F^2
///
/// Uses trace identity to avoid forming the full m×n approximation matrix:
/// ||X - FSG^T||^2 = ||X||^2 - 2*<X*G, F*S> + tr(S^T*F^T*F*S * G^T*G)
pub fn reconstruction_error(
    x: &Array2<f64>,
    f: &Array2<f64>,
    s: &Array2<f64>,
    g: &Array2<f64>,
) -> f64 {
    // Term 1: ||X||_F^2
    let x_norm_sq: f64 = x.iter().map(|&v| v * v).sum();

    // Term 2: <X*G, F*S> = sum of elementwise (X*G) .* (F*S)
    // Both are m×l matrices (l = n_col_clusters, small)
    let xg = x.dot(g); // m×l
    let fs = f.dot(s); // m×l
    let cross: f64 = xg.iter().zip(fs.iter()).map(|(&a, &b)| a * b).sum();

    // Term 3: ||FSG^T||_F^2 = tr(S^T * F^T*F * S * G^T*G)
    // All intermediates are k×k or l×l (tiny)
    let ftf = f.t().dot(f); // k×k
    let gtg = g.t().dot(g); // l×l
    let c = s.t().dot(&ftf).dot(s); // l×l
    let approx_norm_sq: f64 = c.iter().zip(gtg.iter()).map(|(&a, &b)| a * b).sum();

    x_norm_sq - 2.0 * cross + approx_norm_sq
}
