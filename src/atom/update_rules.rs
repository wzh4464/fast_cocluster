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

/// Frobenius norm squared: ||A||_F^2
fn fro_norm_sq(a: &Array2<f64>) -> f64 {
    a.iter().map(|&v| v * v).sum()
}

/// Hadamard (elementwise) inner product: sum(A .* B)
fn hadamard_inner(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// ||FSG^T||_F^2 = tr(S^T * F^T*F * S * G^T*G) via k×k and l×l intermediates
fn approx_fro_norm_sq(f: &Array2<f64>, s: &Array2<f64>, g: &Array2<f64>) -> f64 {
    let ftf = f.t().dot(f); // k×k
    let gtg = g.t().dot(g); // l×l
    let c = s.t().dot(&ftf).dot(s); // l×l
    hadamard_inner(&c, &gtg)
}

/// Frobenius norm squared: ||X - F*S*G^T||_F^2
///
/// Uses trace identity to avoid forming the full m×n approximation matrix:
/// ||X - FSG^T||^2 = ||X||^2 - 2*<X*G, F*S> + ||FSG^T||^2
pub fn reconstruction_error(
    x: &Array2<f64>,
    f: &Array2<f64>,
    s: &Array2<f64>,
    g: &Array2<f64>,
) -> f64 {
    let x_norm_sq = fro_norm_sq(x);

    // Term 2: <X*G, F*S> — both are n×l matrices (l = n_col_clusters, small)
    let xg = x.dot(g); // n×l
    let fs = f.dot(s); // n×l
    let cross = hadamard_inner(&xg, &fs);

    // Term 3: ||FSG^T||_F^2 via small intermediates
    let approx_norm_sq = approx_fro_norm_sq(f, s, g);

    x_norm_sq - 2.0 * cross + approx_norm_sq
}

#[cfg(test)]
mod tests {
    use super::reconstruction_error;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn reconstruction_error_matches_explicit() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let shapes = [(2, 3, 2, 2), (3, 4, 2, 3), (4, 5, 3, 2)];
        for (m, n, k, l) in shapes {
            let x = Array2::from_shape_fn((m, n), |_| rng.gen_range(-1.0..1.0));
            let f = Array2::from_shape_fn((m, k), |_| rng.gen_range(-1.0..1.0));
            let s = Array2::from_shape_fn((k, l), |_| rng.gen_range(-1.0..1.0));
            let g = Array2::from_shape_fn((n, l), |_| rng.gen_range(-1.0..1.0));

            let approx = f.dot(&s).dot(&g.t());
            let diff = &x - &approx;
            let err_explicit: f64 = diff.iter().map(|v| v * v).sum();
            let err_trace = reconstruction_error(&x, &f, &s, &g);

            assert!(
                (err_trace - err_explicit).abs() < 1e-8,
                "mismatch: trace={} explicit={} for (m={},n={},k={},l={})",
                err_trace, err_explicit, m, n, k, l
            );
        }
    }
}
