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
pub fn reconstruction_error(
    x: &Array2<f64>,
    f: &Array2<f64>,
    s: &Array2<f64>,
    g: &Array2<f64>,
) -> f64 {
    let approx = f.dot(s).dot(&g.t());
    let diff = x - &approx;
    diff.mapv(|v| v * v).sum()
}
