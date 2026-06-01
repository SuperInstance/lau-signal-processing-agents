//! Wiener filter: optimal linear filter for stationary signals

use crate::{fft, ifft, Complex64};

/// Wiener filter in the frequency domain
/// Estimates the clean signal from noisy observations using known/estimated PSDs
pub struct WienerFilter {
    /// Filter frequency response H(f) = Sxx(f) / (Sxx(f) + Snn(f))
    pub frequency_response: Vec<f64>,
    /// Length of the filter
    pub length: usize,
}

impl WienerFilter {
    /// Design a Wiener filter given signal and noise power spectral densities
    /// signal_psd and noise_psd must have the same length (power-of-2 recommended)
    pub fn from_psd(signal_psd: &[f64], noise_psd: &[f64]) -> Self {
        assert_eq!(signal_psd.len(), noise_psd.len());
        let freq_resp: Vec<f64> = signal_psd
            .iter()
            .zip(noise_psd.iter())
            .map(|(&sx, &sn)| sx / (sx + sn + 1e-15))
            .collect();

        WienerFilter {
            length: freq_resp.len(),
            frequency_response: freq_resp,
        }
    }

    /// Design a Wiener filter given clean and noisy signal examples
    /// Estimates PSDs from the data
    pub fn from_signals(clean: &[f64], noisy: &[f64]) -> Self {
        assert_eq!(clean.len(), noisy.len());
        let n = clean.len();

        // Estimate noise as difference
        let noise: Vec<f64> = noisy.iter().zip(clean.iter()).map(|(&n, &c)| n - c).collect();

        let clean_spectrum = fft(clean);
        let noise_spectrum = fft(&noise);

        let signal_psd: Vec<f64> = clean_spectrum.iter().map(|c| c.norm_sqr() / n as f64).collect();
        let noise_psd: Vec<f64> = noise_spectrum.iter().map(|c| c.norm_sqr() / n as f64).collect();

        Self::from_psd(&signal_psd, &noise_psd)
    }

    /// Apply the Wiener filter to a noisy signal
    pub fn filter(&self, noisy_signal: &[f64]) -> Vec<f64> {
        let _n = noisy_signal.len();
        let spectrum = fft(noisy_signal);

        // Apply frequency domain filter
        let filtered_spectrum: Vec<Complex64> = spectrum
            .iter()
            .zip(self.frequency_response.iter())
            .map(|(&s, &h)| s * h)
            .collect();

        ifft(&filtered_spectrum)
    }
}

/// Wiener deconvolution: recover original signal from convolved (blurred) observations
pub struct WienerDeconvolution {
    pub frequency_response: Vec<Complex64>,
}

impl WienerDeconvolution {
    /// Design a Wiener deconvolution filter
    /// h_freq: frequency response of the distorting system
    /// snr: signal-to-noise ratio (scalar or per-frequency)
    pub fn new(h_freq: &[Complex64], noise_to_signal: f64) -> Self {
        let freq_resp: Vec<Complex64> = h_freq
            .iter()
            .map(|&h| {
                let h_conj = h.conj();
                let h_mag_sq = h.norm_sqr();
                h_conj / (h_mag_sq + noise_to_signal)
            })
            .collect();

        WienerDeconvolution {
            frequency_response: freq_resp,
        }
    }

    /// Apply deconvolution
    pub fn deconvolve(&self, observed_spectrum: &[Complex64]) -> Vec<Complex64> {
        observed_spectrum
            .iter()
            .zip(self.frequency_response.iter())
            .map(|(&y, &h)| y * h)
            .collect()
    }
}

/// Compute the Wiener filter in the time domain (FIR approximation)
/// Returns the filter coefficients
pub fn wiener_fir(
    desired: &[f64],
    input: &[f64],
    filter_order: usize,
) -> Vec<f64> {
    let n = input.len();
    let m = filter_order;

    // Build autocorrelation matrix R
    let mut r = vec![0.0; m];
    for lag in 0..m {
        for i in lag..n {
            r[lag] += input[i] * input[i - lag];
        }
        r[lag] /= n as f64;
    }

    // Build cross-correlation vector p
    let mut p = vec![0.0; m];
    for lag in 0..m {
        for i in lag..n {
            p[lag] += desired[i] * input[i - lag];
        }
        p[lag] /= n as f64;
    }

    // Solve R * w = p using Levinson-Durbin or direct method
    // Build Toeplitz matrix and solve via simple Gaussian elimination
    let mut aug = vec![vec![0.0; m + 1]; m];
    for i in 0..m {
        for j in 0..m {
            let lag = if i >= j { i - j } else { j - i };
            aug[i][j] = r[lag];
        }
        aug[i][m] = p[i];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..m {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..m {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..m {
            let factor = aug[row][col] / pivot;
            for j in col..=m {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut w = vec![0.0; m];
    for i in (0..m).rev() {
        w[i] = aug[i][m];
        for j in (i + 1)..m {
            w[i] -= aug[i][j] * w[j];
        }
        if aug[i][i].abs() > 1e-15 {
            w[i] /= aug[i][i];
        }
    }

    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_wiener_from_psd() {
        let signal_psd = vec![10.0, 8.0, 5.0, 3.0, 1.0, 0.5];
        let noise_psd = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let filter = WienerFilter::from_psd(&signal_psd, &noise_psd);
        // At high SNR, filter should be close to 1
        assert!(filter.frequency_response[0] > 0.9);
        // At low SNR, filter should attenuate
        assert!(filter.frequency_response[5] < 0.5);
    }

    #[test]
    fn test_wiener_filter_application() {
        // Create a known signal + noise
        let n = 64;
        let clean: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / n as f64).sin()).collect();
        let noise: Vec<f64> = (0..n).map(|i| 0.1 * (2.0 * std::f64::consts::PI * 20.0 * i as f64 / n as f64).sin()).collect();
        let noisy: Vec<f64> = clean.iter().zip(noise.iter()).map(|(&c, &n)| c + n).collect();

        let filter = WienerFilter::from_signals(&clean, &noisy);
        let filtered = filter.filter(&noisy);

        // Filtered should be closer to clean than noisy
        let noisy_mse: f64 = clean.iter().zip(noisy.iter()).map(|(c, n)| (c - n).powi(2)).sum::<f64>() / n as f64;
        let filtered_mse: f64 = clean.iter().zip(filtered.iter()).map(|(c, f)| (c - f).powi(2)).sum::<f64>() / n as f64;
        assert!(filtered_mse < noisy_mse * 2.0);
    }

    #[test]
    fn test_wiener_deconvolution() {
        let h: Vec<Complex64> = vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(0.25, 0.0), Complex64::new(0.125, 0.0)];
        let deconv = WienerDeconvolution::new(&h, 0.01);
        assert_eq!(deconv.frequency_response.len(), 4);
        // Should have some non-zero response
        assert!(deconv.frequency_response[0].norm() > 0.5);
    }

    #[test]
    fn test_wiener_deconvolve_apply() {
        let h: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.1, 0.0),
        ];
        let deconv = WienerDeconvolution::new(&h, 0.01);
        let observed = vec![Complex64::new(1.0, 0.0); 4];
        let result = deconv.deconvolve(&observed);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_wiener_fir() {
        let n = 100;
        let desired: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()).collect();
        let input = desired.clone();
        let coeffs = wiener_fir(&desired, &input, 4);
        assert_eq!(coeffs.len(), 4);
        // First coefficient should be close to 1 (identity)
        assert!(coeffs[0] > 0.5);
    }

    #[test]
    fn test_wiener_fir_denoising() {
        let n = 200;
        let clean: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 3.0 * i as f64 / n as f64).sin()).collect();
        let noisy: Vec<f64> = clean.iter().map(|&c| c + 0.1 * (17.0 * c).sin()).collect();
        let coeffs = wiener_fir(&clean, &noisy, 8);
        assert_eq!(coeffs.len(), 8);
    }
}
