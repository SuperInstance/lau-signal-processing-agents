//! Fourier transform: DFT, FFT (Cooley-Tukey radix-2), frequency analysis

use crate::Complex64;

/// Compute the Discrete Fourier Transform (naive O(n²))
pub fn dft(signal: &[f64]) -> Vec<Complex64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for (t, sample) in signal.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
            sum += Complex64::new(angle.cos(), angle.sin()) * sample;
        }
        result.push(sum);
    }
    result
}

/// Compute the Inverse Discrete Fourier Transform
pub fn idft(spectrum: &[Complex64]) -> Vec<f64> {
    let n = spectrum.len();
    if n == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(n);
    for t in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for (k, coeff) in spectrum.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
            sum += Complex64::new(angle.cos(), angle.sin()) * coeff;
        }
        result.push(sum.re / n as f64);
    }
    result
}

/// Cooley-Tukey radix-2 FFT (in-place, iterative)
/// Input length must be a power of 2.
pub fn fft(signal: &[f64]) -> Vec<Complex64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![Complex64::new(signal[0], 0.0)];
    }
    assert!(n.is_power_of_two(), "FFT requires power-of-2 length, got {}", n);

    // Bit-reversal permutation
    let mut x: Vec<Complex64> = signal.iter().map(|&s| Complex64::new(s, 0.0)).collect();
    let bits = n.trailing_zeros() as usize;

    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            x.swap(i, j);
        }
    }

    // Butterfly operations
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let w = Complex64::new(
                    (angle * k as f64).cos(),
                    (angle * k as f64).sin(),
                );
                let u = x[start + k];
                let v = w * x[start + k + half];
                x[start + k] = u + v;
                x[start + k + half] = u - v;
            }
        }
        len *= 2;
    }

    x
}

/// Inverse FFT using conjugate trick
pub fn ifft(spectrum: &[Complex64]) -> Vec<f64> {
    let n = spectrum.len();
    if n == 0 {
        return vec![];
    }
    let conjugated: Vec<Complex64> = spectrum.iter().map(|c| c.conj()).collect();
    let complex_signal = fft_complex(&conjugated);
    complex_signal.iter().map(|c| c.re / n as f64).collect()
}

/// FFT on complex input
pub fn fft_complex(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    assert!(n.is_power_of_two(), "FFT requires power-of-2 length");

    let mut x = input.to_vec();
    let bits = n.trailing_zeros() as usize;

    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            x.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let w = Complex64::new(
                    (angle * k as f64).cos(),
                    (angle * k as f64).sin(),
                );
                let u = x[start + k];
                let v = w * x[start + k + half];
                x[start + k] = u + v;
                x[start + k + half] = u - v;
            }
        }
        len *= 2;
    }
    x
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Zero-pad signal to next power of 2
pub fn zero_pad_to_power2(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![0.0];
    }
    let m = n.next_power_of_two();
    let mut padded = signal.to_vec();
    padded.resize(m, 0.0);
    padded
}

/// Compute frequency bins for a spectrum of length n sampled at rate fs
pub fn frequency_bins(n: usize, fs: f64) -> Vec<f64> {
    (0..n).map(|k| k as f64 * fs / n as f64).collect()
}

/// Find the dominant frequency in a real-valued signal
pub fn dominant_frequency(signal: &[f64], fs: f64) -> f64 {
    let spectrum = fft(signal);
    let n = signal.len();
    let half = n / 2;
    let mut max_mag = 0.0;
    let mut max_k = 0;
    for k in 1..=half {
        let mag = spectrum[k].norm();
        if mag > max_mag {
            max_mag = mag;
            max_k = k;
        }
    }
    max_k as f64 * fs / n as f64
}

/// Compute magnitude spectrum (one-sided)
pub fn magnitude_spectrum(signal: &[f64]) -> Vec<f64> {
    let spectrum = fft(signal);
    let half = signal.len() / 2 + 1;
    spectrum[..half].iter().map(|c| c.norm() / signal.len() as f64).collect()
}

/// Compute phase spectrum (one-sided)
pub fn phase_spectrum(signal: &[f64]) -> Vec<f64> {
    let spectrum = fft(signal);
    let half = signal.len() / 2 + 1;
    spectrum[..half].iter().map(|c| c.arg()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dft_dc() {
        let signal = vec![1.0; 4];
        let result = dft(&signal);
        assert_abs_diff_eq!(result[0].re, 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dft_sine() {
        let n = 8;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()).collect();
        let result = dft(&signal);
        // Sine at bin 1 should have energy at bin 1
        assert!(result[1].norm() > 2.0);
    }

    #[test]
    fn test_idft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let spectrum = dft(&signal);
        let recovered = idft(&spectrum);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_dc() {
        let signal = vec![1.0; 8];
        let result = fft(&signal);
        assert_abs_diff_eq!(result[0].re, 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fft_matches_dft() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 1.5];
        let dft_result = dft(&signal);
        let fft_result = fft(&signal);
        for (a, b) in dft_result.iter().zip(fft_result.iter()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-10);
            assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ifft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 1.5];
        let spectrum = fft(&signal);
        let recovered = ifft(&spectrum);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_complex() {
        let input = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, -1.0)];
        let result = fft_complex(&input);
        assert_eq!(result.len(), 4);
        assert_abs_diff_eq!(result[0].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_pad() {
        let signal = vec![1.0, 2.0, 3.0];
        let padded = zero_pad_to_power2(&signal);
        assert_eq!(padded.len(), 4);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[3], 0.0);
    }

    #[test]
    fn test_frequency_bins() {
        let bins = frequency_bins(8, 1000.0);
        assert_eq!(bins.len(), 8);
        assert_abs_diff_eq!(bins[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bins[1], 125.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bins[4], 500.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dominant_frequency() {
        let fs = 1000.0;
        let n = 1024;
        let freq = 100.0;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin()).collect();
        let detected = dominant_frequency(&signal, fs);
        assert_abs_diff_eq!(detected, freq, epsilon = fs / n as f64);
    }

    #[test]
    fn test_magnitude_spectrum() {
        let signal = vec![1.0; 16];
        let mag = magnitude_spectrum(&signal);
        assert!(mag[0] > 0.0);
        for &m in &mag[1..] {
            assert_abs_diff_eq!(m, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_phase_spectrum() {
        let signal = vec![1.0; 8];
        let phase = phase_spectrum(&signal);
        assert_abs_diff_eq!(phase[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dft_empty() {
        assert!(dft(&[]).is_empty());
    }

    #[test]
    fn test_fft_empty() {
        assert!(fft(&[]).is_empty());
    }
}
