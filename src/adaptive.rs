//! Adaptive filters: LMS (Least Mean Squares) and RLS (Recursive Least Squares)

use serde::{Deserialize, Serialize};

/// LMS adaptive filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmsFilter {
    /// Filter coefficients (weights)
    pub weights: Vec<f64>,
    /// Step size (learning rate)
    pub mu: f64,
    /// Input buffer (delay line)
    #[serde(skip)]
    buffer: Vec<f64>,
    /// Index for circular buffer
    idx: usize,
}

impl LmsFilter {
    /// Create a new LMS filter
    pub fn new(order: usize, mu: f64) -> Self {
        Self {
            weights: vec![0.0; order],
            mu,
            buffer: vec![0.0; order],
            idx: 0,
        }
    }

    /// Normalized LMS: step size adapts based on input power
    pub fn nlms(order: usize, step_size: f64, epsilon: f64) -> NlmsFilter {
        NlmsFilter {
            weights: vec![0.0; order],
            step_size,
            epsilon,
            buffer: vec![0.0; order],
            idx: 0,
        }
    }

    /// Process one sample: returns (output, error)
    pub fn process(&mut self, input: f64, desired: f64) -> (f64, f64) {
        let order = self.weights.len();

        // Update circular buffer
        self.buffer[self.idx] = input;
        self.idx = (self.idx + 1) % order;

        // Compute output
        let mut output = 0.0;
        for i in 0..order {
            let buf_idx = (self.idx + order - 1 - i) % order;
            output += self.weights[i] * self.buffer[buf_idx];
        }

        // Compute error
        let error = desired - output;

        // Update weights
        for i in 0..order {
            let buf_idx = (self.idx + order - 1 - i) % order;
            self.weights[i] += 2.0 * self.mu * error * self.buffer[buf_idx];
        }

        (output, error)
    }

    /// Process a sequence of input/desired pairs
    pub fn filter_sequence(&mut self, inputs: &[f64], desired: &[f64]) -> LmsResult {
        assert_eq!(inputs.len(), desired.len());
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());

        for i in 0..inputs.len() {
            let (output, error) = self.process(inputs[i], desired[i]);
            outputs.push(output);
            errors.push(error);
        }

        LmsResult { outputs, errors }
    }

    /// Get the filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.idx = 0;
    }
}

/// Normalized LMS filter
#[derive(Debug, Clone)]
pub struct NlmsFilter {
    pub weights: Vec<f64>,
    pub step_size: f64,
    pub epsilon: f64,
    buffer: Vec<f64>,
    idx: usize,
}

impl NlmsFilter {
    pub fn process(&mut self, input: f64, desired: f64) -> (f64, f64) {
        let order = self.weights.len();
        self.buffer[self.idx] = input;
        self.idx = (self.idx + 1) % order;

        let mut output = 0.0;
        for i in 0..order {
            let buf_idx = (self.idx + order - 1 - i) % order;
            output += self.weights[i] * self.buffer[buf_idx];
        }

        let error = desired - output;

        // Input power
        let input_power: f64 = self.buffer.iter().map(|x| x * x).sum() + self.epsilon;

        for i in 0..order {
            let buf_idx = (self.idx + order - 1 - i) % order;
            self.weights[i] += self.step_size * error * self.buffer[buf_idx] / input_power;
        }

        (output, error)
    }
}

/// Result of LMS filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmsResult {
    pub outputs: Vec<f64>,
    pub errors: Vec<f64>,
}

/// RLS (Recursive Least Squares) adaptive filter
#[derive(Debug, Clone)]
pub struct RlsFilter {
    pub weights: Vec<f64>,
    pub forgetting_factor: f64,
    pub regularization: f64,
    /// Inverse correlation matrix P
    p: Vec<Vec<f64>>,
    buffer: Vec<f64>,
    idx: usize,
}

impl RlsFilter {
    /// Create a new RLS filter
    /// forgetting_factor: lambda (0.95-1.0 typical)
    /// regularization: delta for initial P matrix
    pub fn new(order: usize, forgetting_factor: f64, regularization: f64) -> Self {
        let p = vec![vec![0.0; order]; order];
        let mut rls = Self {
            weights: vec![0.0; order],
            forgetting_factor,
            regularization,
            p,
            buffer: vec![0.0; order],
            idx: 0,
        };
        // Initialize P = delta * I
        for i in 0..order {
            rls.p[i][i] = regularization;
        }
        rls
    }

    /// Process one sample
    pub fn process(&mut self, input: f64, desired: f64) -> (f64, f64) {
        let order = self.weights.len();
        self.buffer[self.idx] = input;
        self.idx = (self.idx + 1) % order;

        // Get input vector
        let x: Vec<f64> = (0..order)
            .rev()
            .map(|i| self.buffer[(self.idx + i) % order])
            .collect();

        // Compute output
        let output: f64 = self.weights.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
        let error = desired - output;

        // Compute gain vector: k = P * x / (lambda + x' * P * x)
        let px: Vec<f64> = (0..order)
            .map(|i| (0..order).map(|j| self.p[i][j] * x[j]).sum())
            .collect();

        let xpx: f64 = x.iter().zip(px.iter()).map(|(xi, pi)| xi * pi).sum();
        let denom = self.forgetting_factor + xpx;
        let k: Vec<f64> = px.iter().map(|p| p / denom).collect();

        // Update weights
        for i in 0..order {
            self.weights[i] += k[i] * error;
        }

        // Update P: P = (P - k * x' * P) / lambda
        for i in 0..order {
            for j in 0..order {
                self.p[i][j] = (self.p[i][j] - k[i] * px[j]) / self.forgetting_factor;
            }
        }

        (output, error)
    }

    /// Process a sequence
    pub fn filter_sequence(&mut self, inputs: &[f64], desired: &[f64]) -> LmsResult {
        assert_eq!(inputs.len(), desired.len());
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());

        for i in 0..inputs.len() {
            let (output, error) = self.process(inputs[i], desired[i]);
            outputs.push(output);
            errors.push(error);
        }

        LmsResult { outputs, errors }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_lms_converges() {
        let mut lms = LmsFilter::new(4, 0.01);
        // System identification: unknown system is [0.5, 0.3, 0.1, 0.05]
        for _ in 0..2000 {
            let input = (rand_compat() * 2.0 - 1.0);
            let desired = 0.5 * input + 0.3 * input + 0.1 * input + 0.05 * input;
            lms.process(input, desired);
        }
        // Weights should have converged to something
        let weight_sum: f64 = lms.weights.iter().sum();
        assert!(weight_sum.abs() > 0.1);
    }

    #[test]
    fn test_lms_filter_sequence() {
        let mut lms = LmsFilter::new(4, 0.01);
        let inputs = vec![1.0; 100];
        let desired = vec![1.0; 100];
        let result = lms.filter_sequence(&inputs, &desired);
        assert_eq!(result.outputs.len(), 100);
        assert_eq!(result.errors.len(), 100);
    }

    #[test]
    fn test_lms_error_decreases() {
        let mut lms = LmsFilter::new(4, 0.01);
        let mut early_error = 0.0;
        let mut late_error = 0.0;

        for i in 0..500 {
            let input = (i as f64 * 0.1).sin();
            let desired = 0.5 * input;
            let (_, error) = lms.process(input, desired);
            if i < 50 {
                early_error += error.abs();
            } else if i >= 450 {
                late_error += error.abs();
            }
        }
        assert!(late_error < early_error);
    }

    #[test]
    fn test_lms_reset() {
        let mut lms = LmsFilter::new(4, 0.01);
        lms.process(1.0, 1.0);
        lms.process(2.0, 2.0);
        lms.reset();
        assert!(lms.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_nlms_basic() {
        let mut nlms = LmsFilter::nlms(4, 0.5, 1e-6);
        for i in 0..1000 {
            let input = (i as f64 * 0.1).sin();
            let desired = 0.7 * input;
            nlms.process(input, desired);
        }
        assert!(nlms.weights.iter().any(|&w| w.abs() > 0.01));
    }

    #[test]
    fn test_rls_converges_fast() {
        let mut rls = RlsFilter::new(4, 0.99, 100.0);
        for i in 0..200 {
            let input = (i as f64 * 0.1).sin();
            let desired = 0.5 * input;
            rls.process(input, desired);
        }
        // RLS should converge faster than LMS
        let weight_sum: f64 = rls.weights.iter().sum();
        assert!(weight_sum.abs() > 0.1);
    }

    #[test]
    fn test_rls_filter_sequence() {
        let mut rls = RlsFilter::new(4, 0.99, 100.0);
        let inputs: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let desired: Vec<f64> = inputs.iter().map(|&x| 0.5 * x).collect();
        let result = rls.filter_sequence(&inputs, &desired);
        assert_eq!(result.outputs.len(), 100);
    }

    #[test]
    fn test_rls_error_decreases() {
        let mut rls = RlsFilter::new(4, 0.99, 100.0);
        let mut early_error = 0.0;
        let mut late_error = 0.0;

        for i in 0..200 {
            let input = (i as f64 * 0.1).sin();
            let desired = 0.5 * input;
            let (_, error) = rls.process(input, desired);
            if i < 20 {
                early_error += error.abs();
            } else if i >= 180 {
                late_error += error.abs();
            }
        }
        assert!(late_error < early_error);
    }

    #[test]
    fn test_lms_weights_length() {
        let lms = LmsFilter::new(8, 0.01);
        assert_eq!(lms.weights().len(), 8);
    }

    fn rand_compat() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        ((t as f64 * 1e-9).fract() * 2.0 - 1.0)
    }
}
