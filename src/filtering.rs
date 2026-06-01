//! Filtering: FIR (finite impulse response) and IIR (infinite impulse response) filters

use serde::{Deserialize, Serialize};

/// FIR filter with specified coefficients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirFilter {
    /// Feedforward coefficients (b0, b1, b2, ...)
    pub coeffs: Vec<f64>,
}

impl FirFilter {
    /// Create a new FIR filter with given coefficients
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    /// Design a moving average FIR filter
    pub fn moving_average(order: usize) -> Self {
        let coeff = 1.0 / order as f64;
        Self {
            coeffs: vec![coeff; order],
        }
    }

    /// Design a lowpass FIR filter using windowed sinc method
    pub fn lowpass(cutoff: f64, fs: f64, order: usize) -> Self {
        let fc = cutoff / fs;
        let mut coeffs = Vec::with_capacity(order);
        let mid = (order - 1) as f64 / 2.0;
        for n in 0..order {
            let n_f = n as f64;
            if (n_f - mid).abs() < 1e-10 {
                coeffs.push(2.0 * fc);
            } else {
                let x = std::f64::consts::PI * (n_f - mid);
                coeffs.push((2.0 * fc * x).sin() / x);
            }
        }
        // Apply Hamming window
        let window = crate::windowing::hamming(order);
        for i in 0..order {
            coeffs[i] *= window[i];
        }
        // Normalize
        let sum: f64 = coeffs.iter().sum();
        for c in &mut coeffs {
            *c /= sum;
        }
        Self { coeffs }
    }

    /// Design a highpass FIR filter using spectral inversion
    pub fn highpass(cutoff: f64, fs: f64, order: usize) -> Self {
        let mut lp = Self::lowpass(cutoff, fs, order);
        let mid = order / 2;
        for (i, c) in lp.coeffs.iter_mut().enumerate() {
            if i == mid {
                *c = 1.0 - *c;
            } else {
                *c = -*c;
            }
        }
        lp
    }

    /// Design a bandpass FIR filter
    pub fn bandpass(low_cutoff: f64, high_cutoff: f64, fs: f64, order: usize) -> Self {
        let lp_high = Self::lowpass(high_cutoff, fs, order);
        let lp_low = Self::lowpass(low_cutoff, fs, order);
        let coeffs: Vec<f64> = lp_high.coeffs.iter()
            .zip(lp_low.coeffs.iter())
            .map(|(&h, &l)| h - l)
            .collect();
        Self { coeffs }
    }

    /// Apply the filter to a signal (convolution)
    pub fn filter(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let m = self.coeffs.len();
        let mut output = vec![0.0; n];
        for i in 0..n {
            for j in 0..m {
                if i >= j {
                    output[i] += self.coeffs[j] * signal[i - j];
                }
            }
        }
        output
    }

    /// Get the filter order
    pub fn order(&self) -> usize {
        self.coeffs.len() - 1
    }
}

/// IIR filter (biquad / direct form II)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IirFilter {
    /// Numerator coefficients (b)
    pub b: Vec<f64>,
    /// Denominator coefficients (a), a[0] should be 1.0
    pub a: Vec<f64>,
    /// Internal state (delay line)
    #[serde(skip)]
    state: Vec<f64>,
}

impl IirFilter {
    /// Create a new IIR filter
    pub fn new(b: Vec<f64>, a: Vec<f64>) -> Self {
        assert!(!a.is_empty() && a[0].abs() > 1e-15, "a[0] must be non-zero");
        let state_len = b.len().max(a.len()) - 1;
        Self {
            b,
            a,
            state: vec![0.0; state_len],
        }
    }

    /// Design a first-order lowpass IIR filter
    pub fn lowpass_1st(cutoff: f64, fs: f64) -> Self {
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
        let dt = 1.0 / fs;
        let alpha = dt / (rc + dt);
        Self {
            b: vec![alpha, 0.0],
            a: vec![1.0, alpha - 1.0],
            state: vec![0.0],
        }
    }

    /// Design a first-order highpass IIR filter
    pub fn highpass_1st(cutoff: f64, fs: f64) -> Self {
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
        let dt = 1.0 / fs;
        let alpha = rc / (rc + dt);
        Self {
            b: vec![alpha, -alpha],
            a: vec![1.0, alpha - 1.0],
            state: vec![0.0],
        }
    }

    /// Process a single sample (Direct Form II Transposed)
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let _nb = self.b.len();
        let _na = self.a.len();
        let state_len = self.state.len();

        let output = self.b.get(0).copied().unwrap_or(0.0) * input + self.state[0];

        for i in 0..state_len.saturating_sub(1) {
            self.state[i] = self.b.get(i + 1).copied().unwrap_or(0.0) * input
                - self.a.get(i + 1).copied().unwrap_or(0.0) * output
                + self.state.get(i + 1).copied().unwrap_or(0.0);
        }
        if state_len > 0 {
            let last = state_len - 1;
            self.state[last] = self.b.get(last + 1).copied().unwrap_or(0.0) * input
                - self.a.get(last + 1).copied().unwrap_or(0.0) * output;
        }

        output
    }

    /// Apply the filter to a signal
    pub fn filter(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// Apply a simple first-order exponential smoothing filter
pub fn exponential_smoothing(signal: &[f64], alpha: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(signal.len());
    if signal.is_empty() {
        return result;
    }
    result.push(signal[0]);
    for i in 1..signal.len() {
        result.push(alpha * signal[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

/// Median filter (order N, where N is odd)
pub fn median_filter(signal: &[f64], window_size: usize) -> Vec<f64> {
    assert!(window_size % 2 == 1, "Window size must be odd");
    let half = window_size / 2;
    let n = signal.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let mut window: Vec<f64> = signal[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result[i] = window[window.len() / 2];
    }
    result
}

/// Differentiate signal (first difference)
pub fn differentiate(signal: &[f64]) -> Vec<f64> {
    if signal.len() < 2 {
        return vec![];
    }
    (0..signal.len() - 1).map(|i| signal[i + 1] - signal[i]).collect()
}

/// Integrate signal (cumulative sum)
pub fn integrate(signal: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(signal.len());
    let mut sum = 0.0;
    for &s in signal {
        sum += s;
        result.push(sum);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fir_moving_average() {
        let filter = FirFilter::moving_average(3);
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = filter.filter(&signal);
        // Output at index 2 should be (1+2+3)/3 = 2.0
        assert_abs_diff_eq!(output[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fir_lowpass() {
        let filter = FirFilter::lowpass(100.0, 1000.0, 31);
        assert_eq!(filter.coeffs.len(), 31);
        // Sum of coefficients should be approximately 1
        let sum: f64 = filter.coeffs.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_fir_highpass() {
        let filter = FirFilter::highpass(100.0, 1000.0, 31);
        assert_eq!(filter.coeffs.len(), 31);
    }

    #[test]
    fn test_fir_bandpass() {
        let filter = FirFilter::bandpass(100.0, 300.0, 1000.0, 31);
        assert_eq!(filter.coeffs.len(), 31);
    }

    #[test]
    fn test_fir_identity() {
        let filter = FirFilter::new(vec![1.0]);
        let signal = vec![1.0, 2.0, 3.0];
        let output = filter.filter(&signal);
        for (a, b) in signal.iter().zip(output.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fir_order() {
        let filter = FirFilter::new(vec![0.5, 0.3, 0.2]);
        assert_eq!(filter.order(), 2);
    }

    #[test]
    fn test_iir_lowpass() {
        let mut filter = IirFilter::lowpass_1st(100.0, 1000.0);
        // Step response should approach 1
        let input = vec![1.0; 100];
        let output = filter.filter(&input);
        assert!(output[99] > 0.9);
    }

    #[test]
    fn test_iir_highpass() {
        let mut filter = IirFilter::highpass_1st(100.0, 1000.0);
        // DC should be filtered out
        let input = vec![1.0; 100];
        let output = filter.filter(&input);
        assert!(output[99].abs() < 0.1);
    }

    #[test]
    fn test_iir_reset() {
        let mut filter = IirFilter::new(vec![1.0, 0.5], vec![1.0, -0.3]);
        filter.process_sample(1.0);
        filter.process_sample(2.0);
        filter.reset();
        // After reset, should behave like fresh filter
        let out1 = filter.process_sample(1.0);
        let mut filter2 = IirFilter::new(vec![1.0, 0.5], vec![1.0, -0.3]);
        let out2 = filter2.process_sample(1.0);
        assert_abs_diff_eq!(out1, out2, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_smoothing() {
        let signal = vec![1.0, 1.0, 1.0, 1.0];
        let smoothed = exponential_smoothing(&signal, 0.5);
        assert_eq!(smoothed.len(), 4);
        assert!(smoothed[3] > 0.9);
    }

    #[test]
    fn test_exponential_smoothing_empty() {
        assert!(exponential_smoothing(&[], 0.5).is_empty());
    }

    #[test]
    fn test_median_filter() {
        let signal = vec![1.0, 2.0, 100.0, 4.0, 5.0]; // spike at index 2
        let filtered = median_filter(&signal, 3);
        assert!(filtered[2] < 50.0); // spike should be reduced
    }

    #[test]
    fn test_differentiate() {
        let signal = vec![1.0, 3.0, 6.0, 10.0];
        let diff = differentiate(&signal);
        assert_eq!(diff.len(), 3);
        assert_abs_diff_eq!(diff[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diff[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diff[2], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_differentiate_short() {
        assert!(differentiate(&[]).is_empty());
        assert!(differentiate(&[1.0]).is_empty());
    }

    #[test]
    fn test_integrate() {
        let signal = vec![1.0, 2.0, 3.0];
        let integ = integrate(&signal);
        assert_abs_diff_eq!(integ[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(integ[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(integ[2], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_differentiate_roundtrip() {
        let diff = vec![2.0, 3.0, 4.0];
        let integ = integrate(&diff);
        assert_abs_diff_eq!(integ[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(integ[1], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(integ[2], 9.0, epsilon = 1e-10);
    }
}
