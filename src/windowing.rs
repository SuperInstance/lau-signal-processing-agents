//! Windowing functions: Hamming, Hanning, Blackman, Kaiser

/// Apply a window function to a signal
pub fn apply_window(signal: &[f64], window: &[f64]) -> Vec<f64> {
    signal.iter().zip(window.iter()).map(|(&s, &w)| s * w).collect()
}

/// Generate a Hamming window of length n
pub fn hamming(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

/// Generate a Hanning (Hann) window of length n
pub fn hanning(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

/// Generate a Blackman window of length n
pub fn blackman(n: usize) -> Vec<f64> {
    let a0 = 0.42;
    let a1 = 0.5;
    let a2 = 0.08;
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
            a0 - a1 * x.cos() + a2 * (2.0 * x).cos()
        })
        .collect()
}

/// Generate a Kaiser window of length n with parameter beta
/// Uses the modified Bessel function of the first kind, order 0 (approximation)
pub fn kaiser(n: usize, beta: f64) -> Vec<f64> {
    let alpha = (n - 1) as f64 / 2.0;
    let denom = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let x = (i as f64 - alpha) / alpha;
            let arg = beta * (1.0 - x * x).sqrt();
            bessel_i0(arg) / denom
        })
        .collect()
}

/// Rectangular (uniform) window
pub fn rectangular(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// Flat top window — good for amplitude accuracy
pub fn flat_top(n: usize) -> Vec<f64> {
    let a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368];
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
            a[0] - a[1] * x.cos() + a[2] * (2.0 * x).cos() - a[3] * (3.0 * x).cos() + a[4] * (4.0 * x).cos()
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0
/// Computed via series expansion
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..25 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-16 {
            break;
        }
    }
    sum
}

/// Compute the coherent gain of a window (for normalization)
pub fn coherent_gain(window: &[f64]) -> f64 {
    window.iter().sum::<f64>() / window.len() as f64
}

/// Normalize window so its coherent gain is 1.0
pub fn normalize_window(window: &[f64]) -> Vec<f64> {
    let gain = coherent_gain(window);
    window.iter().map(|&w| w / gain).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_hamming_length() {
        let w = hamming(64);
        assert_eq!(w.len(), 64);
    }

    #[test]
    fn test_hamming_endpoints() {
        let w = hamming(8);
        assert_abs_diff_eq!(w[0], 0.08, epsilon = 0.01);
        assert_abs_diff_eq!(w[7], 0.08, epsilon = 0.01);
    }

    #[test]
    fn test_hamming_peak() {
        let w = hamming(8);
        let max = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > 0.9);
    }

    #[test]
    fn test_hanning_endpoints() {
        let w = hanning(8);
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w[7], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hanning_peak() {
        let w = hanning(8);
        assert_abs_diff_eq!(w[4], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_hanning_sum() {
        let w = hanning(1024);
        let sum: f64 = w.iter().sum();
        // Should be approximately N/2
        assert_abs_diff_eq!(sum, 512.0, epsilon = 1.0);
    }

    #[test]
    fn test_blackman_endpoints() {
        let w = blackman(8);
        assert!(w[0].abs() < 0.01);
        assert!(w[7].abs() < 0.01);
    }

    #[test]
    fn test_blackman_nonneg() {
        let w = blackman(64);
        for &v in &w {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_kaiser_beta_zero() {
        // beta=0 should be rectangular
        let w = kaiser(16, 0.0);
        for &v in &w {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kaiser_sides_smaller_than_center() {
        let w = kaiser(32, 5.0);
        let mid = w.len() / 2;
        assert!(w[mid] > w[0]);
        assert!(w[mid] > w[w.len() - 1]);
    }

    #[test]
    fn test_kaiser_symmetric() {
        let w = kaiser(16, 4.0);
        for i in 0..w.len() / 2 {
            assert_abs_diff_eq!(w[i], w[w.len() - 1 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rectangular() {
        let w = rectangular(10);
        for &v in &w {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_apply_window() {
        let signal = vec![2.0; 4];
        let window = vec![0.5; 4];
        let result = apply_window(&signal, &window);
        for &v in &result {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_coherent_gain() {
        let w = rectangular(10);
        assert_abs_diff_eq!(coherent_gain(&w), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_window() {
        let w = vec![2.0; 4];
        let norm = normalize_window(&w);
        assert_abs_diff_eq!(coherent_gain(&norm), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_flat_top() {
        let w = flat_top(64);
        assert_eq!(w.len(), 64);
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_bessel_i0_zero() {
        assert_abs_diff_eq!(bessel_i0(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bessel_i0_positive() {
        let val = bessel_i0(2.0);
        assert!(val > 1.0);
    }
}
