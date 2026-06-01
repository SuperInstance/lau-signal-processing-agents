//! Kalman filter: optimal estimator for linear Gaussian systems

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Kalman filter state and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter {
    /// State vector (n x 1)
    pub state: Vec<f64>,
    /// State covariance matrix (n x n)
    pub covariance: Vec<Vec<f64>>,
    /// State transition matrix (n x n)
    pub transition: Vec<Vec<f64>>,
    /// Observation matrix (m x n)
    pub observation: Vec<Vec<f64>>,
    /// Process noise covariance (n x n)
    pub process_noise: Vec<Vec<f64>>,
    /// Measurement noise covariance (m x m)
    pub measurement_noise: Vec<Vec<f64>>,
    /// State dimension
    pub n: usize,
    /// Measurement dimension
    pub m: usize,
}

impl KalmanFilter {
    /// Create a new Kalman filter
    pub fn new(
        initial_state: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        transition: Vec<Vec<f64>>,
        observation: Vec<Vec<f64>>,
        process_noise: Vec<Vec<f64>>,
        measurement_noise: Vec<Vec<f64>>,
    ) -> Self {
        let n = initial_state.len();
        let m = measurement_noise.len();
        Self {
            state: initial_state,
            covariance: initial_covariance,
            transition,
            observation,
            process_noise,
            measurement_noise,
            n,
            m,
        }
    }

    /// Create a 1D constant-position Kalman filter
    pub fn constant_position_1d(
        initial_state: f64,
        initial_variance: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Self {
        Self::new(
            vec![initial_state],
            vec![vec![initial_variance]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![process_noise]],
            vec![vec![measurement_noise]],
        )
    }

    /// Create a 1D constant-velocity Kalman filter
    pub fn constant_velocity_1d(
        initial_position: f64,
        initial_velocity: f64,
        process_noise: f64,
        measurement_noise: f64,
        dt: f64,
    ) -> Self {
        Self::new(
            vec![initial_position, initial_velocity],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0, dt], vec![0.0, 1.0]],
            vec![vec![1.0, 0.0]],
            vec![vec![dt.powi(4) / 4.0, dt.powi(3) / 2.0], vec![dt.powi(3) / 2.0, dt.powi(2)]]
                .into_iter()
                .map(|row| row.into_iter().map(|v| v * process_noise).collect())
                .collect(),
            vec![vec![measurement_noise]],
        )
    }

    /// Predict step: project state ahead
    pub fn predict(&mut self) {
        let x = DVector::from_vec(self.state.clone());
        let p = vecs_to_matrix(&self.covariance, self.n, self.n);
        let f = vecs_to_matrix(&self.transition, self.n, self.n);
        let q = vecs_to_matrix(&self.process_noise, self.n, self.n);

        // State prediction: x = F * x
        let x_pred = &f * &x;
        // Covariance prediction: P = F * P * F' + Q
        let p_pred = &f * &p * &f.transpose() + &q;

        self.state = x_pred.iter().cloned().collect();
        self.covariance = matrix_to_vecs(&p_pred, self.n, self.n);
    }

    /// Update step: incorporate measurement
    pub fn update(&mut self, measurement: &[f64]) {
        let x = DVector::from_vec(self.state.clone());
        let p = vecs_to_matrix(&self.covariance, self.n, self.n);
        let h = vecs_to_matrix(&self.observation, self.m, self.n);
        let r = vecs_to_matrix(&self.measurement_noise, self.m, self.m);
        let z = DVector::from_vec(measurement.to_vec());

        // Innovation: y = z - H * x
        let y = &z - &(&h * &x);

        // Innovation covariance: S = H * P * H' + R
        let s = &h * &p * &h.transpose() + &r;

        // Kalman gain: K = P * H' * S^-1
        let s_inv = s.clone().try_inverse().expect("Innovation covariance is singular");
        let k = &p * &h.transpose() * &s_inv;

        // State update: x = x + K * y
        let x_updated = &x + &(&k * &y);

        // Covariance update: P = (I - K * H) * P
        let identity = DMatrix::identity(self.n, self.n);
        let p_updated = (&identity - &k * &h) * &p;

        self.state = x_updated.iter().cloned().collect();
        self.covariance = matrix_to_vecs(&p_updated, self.n, self.n);
    }

    /// Run full predict-update cycle
    pub fn step(&mut self, measurement: &[f64]) {
        self.predict();
        self.update(measurement);
    }

    /// Process a sequence of measurements
    pub fn filter_sequence(&mut self, measurements: &[Vec<f64>]) -> Vec<Vec<f64>> {
        measurements
            .iter()
            .map(|m| {
                self.step(m);
                self.state.clone()
            })
            .collect()
    }

    /// Get the current state estimate
    pub fn state_estimate(&self) -> &[f64] {
        &self.state
    }

    /// Get the current state uncertainty (trace of covariance)
    pub fn uncertainty(&self) -> f64 {
        self.covariance.iter().enumerate().map(|(i, row)| row[i]).sum()
    }
}

fn matrix_to_vecs(m: &DMatrix<f64>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| (0..cols).map(|j| m[(i, j)]).collect())
        .collect()
}

/// Convert row-major Vec<Vec<f64>> to nalgebra DMatrix (column-major)
fn vecs_to_matrix(vecs: &[Vec<f64>], rows: usize, cols: usize) -> DMatrix<f64> {
    let mut data = vec![0.0; rows * cols];
    for (i, row) in vecs.iter().enumerate().take(rows) {
        for (j, &val) in row.iter().enumerate().take(cols) {
            data[j * rows + i] = val; // column-major
        }
    }
    DMatrix::from_vec(rows, cols, data)
}

/// Extended Kalman Filter (EKF) for nonlinear systems
#[derive(Debug, Clone)]
pub struct ExtendedKalmanFilter {
    pub state: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub process_noise: Vec<Vec<f64>>,
    pub measurement_noise: Vec<Vec<f64>>,
    pub n: usize,
    pub m: usize,
}

impl ExtendedKalmanFilter {
    pub fn new(
        initial_state: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        process_noise: Vec<Vec<f64>>,
        measurement_noise: Vec<Vec<f64>>,
    ) -> Self {
        let n = initial_state.len();
        let m = measurement_noise.len();
        Self {
            state: initial_state,
            covariance: initial_covariance,
            process_noise,
            measurement_noise,
            n,
            m,
        }
    }

    /// EKF step with user-provided dynamics and observation functions + Jacobians
    pub fn step<F, H, FJ, HJ>(
        &mut self,
        measurement: &[f64],
        f: F,
        h: H,
        fj: FJ,
        hj: HJ,
    ) where
        F: Fn(&[f64]) -> Vec<f64>,
        H: Fn(&[f64]) -> Vec<f64>,
        FJ: Fn(&[f64]) -> Vec<Vec<f64>>,
        HJ: Fn(&[f64]) -> Vec<Vec<f64>>,
    {
        // Predict
        let x_pred = f(&self.state);
        let f_jac = fj(&self.state);
        let f_mat = vecs_to_matrix(&f_jac, self.n, self.n);
        let p = vecs_to_matrix(&self.covariance, self.n, self.n);
        let q = vecs_to_matrix(&self.process_noise, self.n, self.n);

        let p_pred = &f_mat * &p * &f_mat.transpose() + &q;

        // Update
        let z_pred = h(&x_pred);
        let h_jac = hj(&x_pred);
        let h_mat = vecs_to_matrix(&h_jac, self.m, self.n);
        let r = vecs_to_matrix(&self.measurement_noise, self.m, self.m);
        let z = DVector::from_vec(measurement.to_vec());
        let z_p = DVector::from_vec(z_pred);
        let x_p = DVector::from_vec(x_pred);

        let y = &z - &z_p;
        let s = &h_mat * &p_pred * &h_mat.transpose() + &r;
        let s_inv = s.clone().try_inverse().expect("Singular innovation covariance");
        let k = &p_pred * &h_mat.transpose() * &s_inv;

        let x_updated = &x_p + &(&k * &y);
        let identity = DMatrix::identity(self.n, self.n);
        let p_updated = (&identity - &k * &h_mat) * &p_pred;

        self.state = x_updated.iter().cloned().collect();
        self.covariance = matrix_to_vecs(&p_updated, self.n, self.n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_kf_constant_position_converges() {
        let mut kf = KalmanFilter::constant_position_1d(0.0, 100.0, 0.01, 1.0);
        for _ in 0..50 {
            kf.step(&[5.0]);
        }
        assert_abs_diff_eq!(kf.state[0], 5.0, epsilon = 0.5);
    }

    #[test]
    fn test_kf_constant_position_reduces_variance() {
        let mut kf = KalmanFilter::constant_position_1d(0.0, 100.0, 0.01, 1.0);
        let initial_uncertainty = kf.uncertainty();
        for _ in 0..20 {
            kf.step(&[5.0]);
        }
        assert!(kf.uncertainty() < initial_uncertainty);
    }

    #[test]
    fn test_kf_constant_velocity_tracking() {
        let mut kf = KalmanFilter::constant_velocity_1d(0.0, 1.0, 0.1, 1.0, 1.0);
        // Simulate constant velocity of 2 units/step
        for t in 0..100 {
            let true_pos = 2.0 * t as f64;
            kf.step(&[true_pos + 0.1 * (t as f64 * 3.7).sin()]); // noisy measurement
        }
        // Velocity estimate should be close to 2.0
        assert_abs_diff_eq!(kf.state[1], 2.0, epsilon = 1.0);
    }

    #[test]
    fn test_kf_predict_only() {
        let mut kf = KalmanFilter::constant_position_1d(10.0, 1.0, 0.01, 1.0);
        kf.predict();
        // State should stay the same for constant position model
        assert_abs_diff_eq!(kf.state[0], 10.0, epsilon = 1e-10);
        // Variance should increase
        assert!(kf.uncertainty() > 1.0);
    }

    #[test]
    fn test_kf_filter_sequence() {
        let mut kf = KalmanFilter::constant_position_1d(0.0, 100.0, 0.01, 1.0);
        let measurements: Vec<Vec<f64>> = (0..20).map(|_| vec![5.0]).collect();
        let states = kf.filter_sequence(&measurements);
        assert_eq!(states.len(), 20);
        assert_abs_diff_eq!(states[19][0], 5.0, epsilon = 0.5);
    }

    #[test]
    fn test_kf_state_estimate() {
        let kf = KalmanFilter::constant_position_1d(3.0, 1.0, 0.01, 1.0);
        assert_eq!(kf.state_estimate(), &[3.0]);
    }

    #[test]
    fn test_kf_uncertainty() {
        let kf = KalmanFilter::constant_position_1d(0.0, 5.0, 0.01, 1.0);
        assert_abs_diff_eq!(kf.uncertainty(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ekf_nonlinear_tracking() {
        let mut ekf = ExtendedKalmanFilter::new(
            vec![0.0, 0.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0.1, 0.0], vec![0.0, 0.1]],
            vec![vec![1.0]],
        );

        // Nonlinear: state = [position, velocity], observe position through sqrt
        for t in 0..30 {
            let true_pos = 2.0 * t as f64;
            ekf.step(
                &[true_pos.sqrt()],
                |x| vec![x[0] + x[1], x[1]], // f: constant velocity
                |x| vec![x[0].sqrt()],         // h: observe sqrt of position
                |_x| vec![vec![1.0, 1.0], vec![0.0, 1.0]], // F Jacobian
                |x| vec![vec![0.5 / x[0].sqrt().max(0.01), 0.0]], // H Jacobian
            );
        }
        // Should track reasonably
        assert!(ekf.state[0] > 40.0);
    }
}
