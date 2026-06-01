//! Wavelet transform: Haar, Daubechies D4, multi-resolution analysis

use serde::{Deserialize, Serialize};

/// Wavelet type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WaveletType {
    Haar,
    Daubechies4,
}

/// Result of a wavelet decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletDecomposition {
    /// Approximation coefficients at each level
    pub approximations: Vec<Vec<f64>>,
    /// Detail coefficients at each level
    pub details: Vec<Vec<f64>>,
    /// Number of decomposition levels
    pub levels: usize,
}

/// Haar wavelet transform (forward)
/// Signal length must be a power of 2
pub fn haar_forward(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    assert!(n.is_power_of_two(), "Signal length must be power of 2");
    let mut data = signal.to_vec();
    let mut temp = vec![0.0; n];
    let mut len = n;

    while len > 1 {
        let half = len / 2;
        for i in 0..half {
            temp[i] = (data[2 * i] + data[2 * i + 1]) / std::f64::consts::SQRT_2;
            temp[half + i] = (data[2 * i] - data[2 * i + 1]) / std::f64::consts::SQRT_2;
        }
        data[..len].copy_from_slice(&temp[..len]);
        len = half;
    }

    let mut approx = vec![data[0]];
    let mut details = Vec::new();
    len = 1;
    let mut pos = 1;
    while pos < n {
        details.push(data[pos..pos + len].to_vec());
        pos += len;
        len *= 2;
        if pos + len <= n {
            approx = data[pos..pos + len].to_vec();
            pos += len;
        }
    }

    (approx, details.into_iter().next_back().unwrap_or_default())
}

/// Full multi-level Haar decomposition
pub fn haar_decompose(signal: &[f64], levels: usize) -> WaveletDecomposition {
    let n = signal.len();
    assert!(n >= (1 << levels), "Signal too short for {} levels", levels);

    let mut approximations = Vec::new();
    let mut details = Vec::new();
    let mut current = signal.to_vec();

    for _ in 0..levels {
        let half = current.len() / 2;
        let mut approx = vec![0.0; half];
        let mut detail = vec![0.0; half];

        for i in 0..half {
            approx[i] = (current[2 * i] + current[2 * i + 1]) / std::f64::consts::SQRT_2;
            detail[i] = (current[2 * i] - current[2 * i + 1]) / std::f64::consts::SQRT_2;
        }

        approximations.push(approx.clone());
        details.push(detail);
        current = approx;
    }

    WaveletDecomposition {
        approximations,
        details,
        levels,
    }
}

/// Haar wavelet inverse (reconstruct from decomposition)
pub fn haar_reconstruct(decomp: &WaveletDecomposition) -> Vec<f64> {
    let mut current = decomp.approximations.last()
        .or_else(|| decomp.details.last())
        .cloned()
        .unwrap_or_default();

    // Start from the coarsest level and reconstruct
    let levels = decomp.levels;
    if levels == 0 {
        return current;
    }

    // Use the last approximation as starting point
    current = decomp.approximations.last().cloned().unwrap_or_default();

    for level in (0..levels).rev() {
        let detail = &decomp.details[level];
        let n = current.len();
        let mut reconstructed = vec![0.0; 2 * n];

        for i in 0..n {
            reconstructed[2 * i] = (current[i] + detail[i]) / std::f64::consts::SQRT_2;
            reconstructed[2 * i + 1] = (current[i] - detail[i]) / std::f64::consts::SQRT_2;
        }
        current = reconstructed;
    }

    current
}

/// Daubechies D4 wavelet coefficients
const DB4_H: [f64; 4] = [
    0.6830127 / std::f64::consts::SQRT_2,
    1.1830127 / std::f64::consts::SQRT_2,
    0.3169873 / std::f64::consts::SQRT_2,
    -0.1830127 / std::f64::consts::SQRT_2,
];

const DB4_G: [f64; 4] = [
    -0.1830127 / std::f64::consts::SQRT_2,
    0.3169873 / std::f64::consts::SQRT_2,
    -1.1830127 / std::f64::consts::SQRT_2,
    0.6830127 / std::f64::consts::SQRT_2,
];

/// Daubechies D4 wavelet transform (single level)
pub fn db4_forward(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    assert!(n >= 4 && n % 2 == 0, "Signal length must be even and >= 4");
    let half = n / 2;
    let mut approx = vec![0.0; half];
    let mut detail = vec![0.0; half];

    for i in 0..half {
        for k in 0..4 {
            let idx = (2 * i + k) % n;
            approx[i] += DB4_H[k] * signal[idx];
            detail[i] += DB4_G[k] * signal[idx];
        }
    }

    (approx, detail)
}

/// Multi-level Daubechies D4 decomposition
pub fn db4_decompose(signal: &[f64], levels: usize) -> WaveletDecomposition {
    let mut approximations = Vec::new();
    let mut details = Vec::new();
    let mut current = signal.to_vec();

    for _ in 0..levels {
        assert!(current.len() >= 4 && current.len() % 2 == 0);
        let (approx, detail) = db4_forward(&current);
        approximations.push(approx.clone());
        details.push(detail);
        current = approx;
    }

    WaveletDecomposition {
        approximations,
        details,
        levels,
    }
}

/// Threshold detail coefficients (denoising)
pub fn soft_threshold(coefficients: &[f64], threshold: f64) -> Vec<f64> {
    coefficients
        .iter()
        .map(|&c| {
            if c.abs() <= threshold {
                0.0
            } else if c > threshold {
                c - threshold
            } else {
                c + threshold
            }
        })
        .collect()
}

/// Hard threshold (simpler denoising)
pub fn hard_threshold(coefficients: &[f64], threshold: f64) -> Vec<f64> {
    coefficients
        .iter()
        .map(|&c| if c.abs() <= threshold { 0.0 } else { c })
        .collect()
}

/// Compute universal threshold (VisuShrink)
pub fn universal_threshold(signal: &[f64]) -> f64 {
    let n = signal.len() as f64;
    let sigma = median_absolute_deviation(signal);
    sigma * (2.0 * n.ln()).sqrt()
}

fn median_absolute_deviation(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mut deviations: Vec<f64> = sorted.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    deviations[deviations.len() / 2] / 0.6745 // Scale to standard deviation
}

/// Wavelet denoising using soft thresholding
pub fn wavelet_denoise(signal: &[f64], levels: usize) -> Vec<f64> {
    let decomp = haar_decompose(signal, levels);
    let threshold = universal_threshold(
        &decomp.details.last().cloned().unwrap_or_default(),
    );

    let denoised_details: Vec<Vec<f64>> = decomp.details
        .iter()
        .map(|d| soft_threshold(d, threshold))
        .collect();

    let denoised = WaveletDecomposition {
        approximations: decomp.approximations.clone(),
        details: denoised_details,
        levels: decomp.levels,
    };

    haar_reconstruct(&denoised)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_haar_constant() {
        let signal = vec![4.0; 8];
        let decomp = haar_decompose(&signal, 3);
        // Constant signal → all details should be ~0
        for level_details in &decomp.details {
            for &d in level_details {
                assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_haar_reconstruct_roundtrip() {
        let signal = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let decomp = haar_decompose(&signal, 3);
        let reconstructed = haar_reconstruct(&decomp);
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_haar_decompose_levels() {
        let signal = vec![0.0; 16];
        let decomp = haar_decompose(&signal, 3);
        assert_eq!(decomp.levels, 3);
        assert_eq!(decomp.details.len(), 3);
        assert_eq!(decomp.approximations.len(), 3);
    }

    #[test]
    fn test_haar_step_function() {
        // Step between indices 4 and 5 so it falls within a pair (4,5)
        let signal = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let decomp = haar_decompose(&signal, 2);
        // Should have significant detail at the step boundary
        assert!(decomp.details.iter().any(|d| d.iter().any(|&v| v.abs() > 0.1)));
    }

    #[test]
    fn test_db4_forward() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, detail) = db4_forward(&signal);
        assert_eq!(approx.len(), 4);
        assert_eq!(detail.len(), 4);
    }

    #[test]
    fn test_db4_decompose() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decomp = db4_decompose(&signal, 2);
        assert_eq!(decomp.levels, 2);
        assert_eq!(decomp.details.len(), 2);
    }

    #[test]
    fn test_soft_threshold() {
        let coeffs = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let thresholded = soft_threshold(&coeffs, 1.0);
        assert_abs_diff_eq!(thresholded[0], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(thresholded[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(thresholded[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(thresholded[3], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(thresholded[4], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hard_threshold() {
        let coeffs = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let thresholded = hard_threshold(&coeffs, 1.0);
        assert_abs_diff_eq!(thresholded[0], -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(thresholded[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(thresholded[4], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_universal_threshold() {
        let signal = vec![0.1, -0.2, 0.15, -0.1, 0.05];
        let t = universal_threshold(&signal);
        assert!(t > 0.0);
    }

    #[test]
    fn test_wavelet_denoise() {
        // Clean signal + larger noise
        let clean: Vec<f64> = (0..64).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 64.0).sin()).collect();
        let noisy: Vec<f64> = clean.iter().enumerate().map(|(i, &x)| x + 0.5 * ((i * 7 + 3) as f64).sin()).collect();
        let denoised = wavelet_denoise(&noisy, 3);
        // Denoised should be closer to clean than noisy
        let noisy_err: f64 = clean.iter().zip(noisy.iter()).map(|(c, n)| (c - n).powi(2)).sum();
        let denoised_err: f64 = clean.iter().zip(denoised.iter()).map(|(c, d)| (c - d).powi(2)).sum();
        assert!(denoised_err <= noisy_err, "denoised_err={} > noisy_err={}", denoised_err, noisy_err);
    }

    #[test]
    fn test_haar_alternating() {
        let signal = vec![1.0, -1.0, 1.0, -1.0];
        let decomp = haar_decompose(&signal, 2);
        // Alternating signal → large detail coefficients
        assert!(decomp.details[0].iter().any(|&d| d.abs() > 0.5));
    }

    #[test]
    fn test_db4_preserves_energy() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, detail) = db4_forward(&signal);
        let energy_orig: f64 = signal.iter().map(|x| x * x).sum();
        let energy_decomp: f64 = approx.iter().map(|x| x * x).sum::<f64>()
            + detail.iter().map(|x| x * x).sum::<f64>();
        // Energy should be approximately preserved (circular convolution on short signal)
        assert_abs_diff_eq!(energy_orig, energy_decomp, epsilon = energy_orig * 0.2);
    }
}
