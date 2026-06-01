//! Power spectral density: Welch's method, periodogram

use crate::fft;
use crate::windowing;

/// Compute the periodogram (non-parametric PSD estimate)
/// Returns one-sided PSD: P[k] for k = 0..N/2+1
pub fn periodogram(signal: &[f64], fs: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let spectrum = fft(signal);
    let half = n / 2 + 1;
    let mut psd = Vec::with_capacity(half);
    for k in 0..half {
        let mag_sq = spectrum[k].norm_sqr();
        psd.push(mag_sq / (fs * n as f64));
    }
    // Double the interior bins (one-sided correction)
    for k in 1..half - 1 {
        psd[k] *= 2.0;
    }
    psd
}

/// Welch's method for PSD estimation
/// Divides signal into overlapping segments, windows each, averages periodograms
pub fn welch(
    signal: &[f64],
    fs: f64,
    segment_length: usize,
    overlap: usize,
    window_type: WindowType,
) -> WelchResult {
    let n = signal.len();
    assert!(segment_length > 0 && segment_length <= n, "Invalid segment length");
    assert!(overlap < segment_length, "Overlap must be less than segment length");

    let step = segment_length - overlap;
    let window = match window_type {
        WindowType::Hanning => windowing::hanning(segment_length),
        WindowType::Hamming => windowing::hamming(segment_length),
        WindowType::Blackman => windowing::blackman(segment_length),
        WindowType::Rectangular => windowing::rectangular(segment_length),
    };

    // Window normalization factor
    let w_norm: f64 = window.iter().map(|&w| w * w).sum();

    // Collect segments
    let mut segments = Vec::new();
    let mut start = 0;
    while start + segment_length <= n {
        let segment = &signal[start..start + segment_length];
        let windowed = windowing::apply_window(segment, &window);
        segments.push(windowed);
        start += step;
    }

    let num_segments = segments.len();
    assert!(num_segments > 0, "No segments produced");

    let half = segment_length / 2 + 1;
    let mut psd = vec![0.0; half];

    for seg in &segments {
        let spectrum = fft(seg);
        for k in 0..half {
            psd[k] += spectrum[k].norm_sqr();
        }
    }

    // Normalize
    let scale = fs * w_norm * num_segments as f64;
    for k in 0..half {
        psd[k] /= scale;
    }

    // One-sided correction
    for k in 1..half - 1 {
        psd[k] *= 2.0;
    }

    let freqs: Vec<f64> = (0..half)
        .map(|k| k as f64 * fs / segment_length as f64)
        .collect();

    WelchResult {
        frequencies: freqs,
        psd,
        num_segments,
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum WindowType {
    Hanning,
    Hamming,
    Blackman,
    Rectangular,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WelchResult {
    pub frequencies: Vec<f64>,
    pub psd: Vec<f64>,
    pub num_segments: usize,
}

/// Compute the total power in a PSD estimate
pub fn total_power(psd: &[f64], freq_resolution: f64) -> f64 {
    psd.iter().sum::<f64>() * freq_resolution
}

/// Find peak frequency from PSD
pub fn peak_frequency(frequencies: &[f64], psd: &[f64]) -> f64 {
    let mut max_psd = f64::NEG_INFINITY;
    let mut max_idx = 0;
    for (i, &p) in psd.iter().enumerate() {
        if p > max_psd {
            max_psd = p;
            max_idx = i;
        }
    }
    frequencies[max_idx]
}

/// Compute the spectrogram (time-frequency representation)
/// Returns a matrix where each row is a PSD estimate for a time segment
pub fn spectrogram(
    signal: &[f64],
    fs: f64,
    segment_length: usize,
    overlap: usize,
    window_type: WindowType,
) -> SpectrogramResult {
    let welch_result = welch(signal, fs, segment_length, overlap, window_type);
    let step = segment_length - overlap;
    let n = signal.len();
    let mut times = Vec::new();
    let mut rows = Vec::new();

    let window = match window_type {
        WindowType::Hanning => windowing::hanning(segment_length),
        WindowType::Hamming => windowing::hamming(segment_length),
        WindowType::Blackman => windowing::blackman(segment_length),
        WindowType::Rectangular => windowing::rectangular(segment_length),
    };
    let w_norm: f64 = window.iter().map(|&w| w * w).sum();
    let scale = fs * w_norm;

    let half = segment_length / 2 + 1;
    let mut start = 0;
    while start + segment_length <= n {
        let segment = &signal[start..start + segment_length];
        let windowed = windowing::apply_window(segment, &window);
        let spectrum = fft(&windowed);
        let mut psd = vec![0.0; half];
        for k in 0..half {
            psd[k] = spectrum[k].norm_sqr() / scale;
        }
        for k in 1..half - 1 {
            psd[k] *= 2.0;
        }
        times.push((start + segment_length / 2) as f64 / fs);
        rows.push(psd);
        start += step;
    }

    SpectrogramResult {
        frequencies: welch_result.frequencies,
        times,
        matrix: rows,
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectrogramResult {
    pub frequencies: Vec<f64>,
    pub times: Vec<f64>,
    pub matrix: Vec<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_periodogram_dc() {
        let signal = vec![1.0; 64];
        let psd = periodogram(&signal, 1000.0);
        assert!(psd[0] > 0.0);
        // DC bin should dominate
        for &p in &psd[1..] {
            assert!(psd[0] > p * 100.0);
        }
    }

    #[test]
    fn test_periodogram_length() {
        let signal = vec![0.0; 128];
        let psd = periodogram(&signal, 1000.0);
        assert_eq!(psd.len(), 65); // 128/2 + 1
    }

    #[test]
    fn test_periodogram_sine() {
        let fs = 1000.0;
        let n = 256;
        let freq = 100.0;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin()).collect();
        let psd = periodogram(&signal, fs);
        let peak_bin = (freq / (fs / n as f64)) as usize;
        assert!(psd[peak_bin] > psd[0]);
    }

    #[test]
    fn test_welch_basic() {
        let signal = vec![1.0; 256];
        let result = welch(&signal, 1000.0, 64, 32, WindowType::Hanning);
        assert!(result.psd[0] > 0.0);
        assert!(result.num_segments > 1);
    }

    #[test]
    fn test_welch_num_segments() {
        let signal = vec![0.0; 128];
        let result = welch(&signal, 1000.0, 64, 32, WindowType::Hanning);
        // 128 / (64 - 32) = 4 segments (positions 0, 32, 64, 96)
        assert_eq!(result.num_segments, 4);
    }

    #[test]
    fn test_welch_frequencies() {
        let signal = vec![0.0; 256];
        let result = welch(&signal, 1000.0, 128, 64, WindowType::Hamming);
        assert_abs_diff_eq!(result.frequencies[0], 0.0, epsilon = 1e-10);
        let freq_res = 1000.0 / 128.0;
        assert_abs_diff_eq!(result.frequencies[1], freq_res, epsilon = 1e-10);
    }

    #[test]
    fn test_welch_sine_peak() {
        let fs = 1000.0;
        let n = 512;
        let freq = 50.0;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin()).collect();
        let result = welch(&signal, fs, 128, 64, WindowType::Hanning);
        let peak = peak_frequency(&result.frequencies, &result.psd);
        assert_abs_diff_eq!(peak, freq, epsilon = 1000.0 / 128.0);
    }

    #[test]
    fn test_total_power() {
        let psd = vec![1.0; 10];
        let power = total_power(&psd, 1.0);
        assert_abs_diff_eq!(power, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectrogram() {
        let fs = 1000.0;
        let n = 256;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 100.0 * i as f64 / fs).sin()).collect();
        let result = spectrogram(&signal, fs, 64, 32, WindowType::Hanning);
        assert!(!result.matrix.is_empty());
        assert!(!result.times.is_empty());
        assert_eq!(result.matrix[0].len(), 33); // 64/2 + 1
    }

    #[test]
    fn test_periodogram_empty() {
        assert!(periodogram(&[], 1000.0).is_empty());
    }

    #[test]
    fn test_welch_window_types() {
        let signal = vec![1.0; 256];
        for wt in [WindowType::Hanning, WindowType::Hamming, WindowType::Blackman, WindowType::Rectangular] {
            let result = welch(&signal, 1000.0, 64, 32, wt);
            assert!(!result.psd.is_empty());
        }
    }
}
