//! Compressed sensing: recover sparse signals from few observations


/// Compressed sensing reconstruction using Iterative Hard Thresholding (IHT)
/// Recovers a sparse signal from undersampled measurements
pub fn iterative_hard_thresholding(
    measurements: &[f64],
    sensing_matrix: &[Vec<f64>],  // m x n (m << n)
    sparsity: usize,
    max_iterations: usize,
    tolerance: f64,
) -> Vec<f64> {
    let _m = measurements.len();
    let n = sensing_matrix[0].len();

    let mut x = vec![0.0; n];

    // Compute step size (upper bound: 1 / largest singular value of A'A)
    // Approximate as 1 / (max row norm squared)
    let mut step: f64 = 0.0;
    for row in sensing_matrix {
        let norm_sq: f64 = row.iter().map(|v| v * v).sum();
        step = step.max(norm_sq);
    }
    step = 1.0 / step;

    for _ in 0..max_iterations {
        // Compute residual: r = y - A * x
        let ax = mat_vec_mul(sensing_matrix, &x);
        let residual: Vec<f64> = measurements.iter().zip(ax.iter()).map(|(&y, &axi)| y - axi).collect();

        // Gradient: A' * r
        let gradient = mat_vec_mul_transpose(sensing_matrix, &residual);

        // Update: x = x + step * gradient
        for i in 0..n {
            x[i] += step * gradient[i];
        }

        // Hard threshold: keep only top-k entries
        hard_threshold_keep(&mut x, sparsity);

        // Check convergence
        let ax_new = mat_vec_mul(sensing_matrix, &x);
        let error: f64 = measurements.iter().zip(ax_new.iter())
            .map(|(y, axi)| (y - axi).powi(2))
            .sum::<f64>()
            .sqrt();

        if error < tolerance {
            break;
        }
    }

    x
}

/// Orthogonal Matching Pursuit (OMP) for sparse recovery
pub fn orthogonal_matching_pursuit(
    measurements: &[f64],
    dictionary: &[Vec<f64>],  // m x n
    sparsity: usize,
) -> OmpResult {
    let m = measurements.len();
    let n = dictionary[0].len();
    let k = sparsity.min(n);

    let mut support: Vec<usize> = Vec::new();
    let mut residual = measurements.to_vec();
    let mut coefficients = vec![0.0; n];

    for _ in 0..k {
        // Find the atom most correlated with the residual
        let mut best_idx = 0;
        let mut best_corr = f64::NEG_INFINITY;
        for j in 0..n {
            if support.contains(&j) {
                continue;
            }
            let col: Vec<f64> = dictionary.iter().map(|row| row[j]).collect();
            let corr: f64 = col.iter().zip(residual.iter()).map(|(a, r)| a * r).sum();
            if corr.abs() > best_corr {
                best_corr = corr.abs();
                best_idx = j;
            }
        }

        support.push(best_idx);

        // Least squares solve for supported coefficients
        // Build sub-matrix A_s
        let a_s: Vec<Vec<f64>> = support.iter()
            .map(|&idx| dictionary.iter().map(|row| row[idx]).collect())
            .collect();

        // Solve a_s^T * a_s * c = a_s^T * y
        let cols = a_s.len();
        let mut ata = vec![vec![0.0; cols]; cols];
        for i in 0..cols {
            for j in 0..cols {
                ata[i][j] = a_s[i].iter().zip(a_s[j].iter()).map(|(a, b)| a * b).sum();
            }
        }

        let aty: Vec<f64> = a_s.iter()
            .map(|col| col.iter().zip(measurements.iter()).map(|(a, y)| a * y).sum())
            .collect();

        let c = solve_symmetric(&ata, &aty);

        // Update coefficients and residual
        for (i, &idx) in support.iter().enumerate() {
            coefficients[idx] = c[i];
        }

        let approx: Vec<f64> = (0..m)
            .map(|row_idx| {
                support.iter().zip(c.iter())
                    .map(|(&col_idx, &ci)| dictionary[row_idx][col_idx] * ci)
                    .sum()
            })
            .collect();

        residual = measurements.iter().zip(approx.iter()).map(|(y, a)| y - a).collect();
    }

    OmpResult {
        signal: coefficients,
        support,
        residual_norm: residual.iter().map(|r| r * r).sum::<f64>().sqrt(),
    }
}

/// Result of OMP reconstruction
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OmpResult {
    pub signal: Vec<f64>,
    pub support: Vec<usize>,
    pub residual_norm: f64,
}

/// Basis Pursuit Denoising using iterative soft thresholding (ISTA)
pub fn basis_pursuit_denoising(
    measurements: &[f64],
    sensing_matrix: &[Vec<f64>],
    lambda: f64,
    max_iterations: usize,
) -> Vec<f64> {
    let n = sensing_matrix[0].len();
    let mut x = vec![0.0; n];

    // Step size
    let mut lip: f64 = 0.0;
    for row in sensing_matrix {
        let norm_sq: f64 = row.iter().map(|v| v * v).sum();
        lip = lip.max(norm_sq);
    }
    let step = 1.0 / lip;

    for _ in 0..max_iterations {
        let ax = mat_vec_mul(sensing_matrix, &x);
        let residual: Vec<f64> = measurements.iter().zip(ax.iter()).map(|(&y, &axi)| y - axi).collect();
        let gradient = mat_vec_mul_transpose(sensing_matrix, &residual);

        for i in 0..n {
            x[i] = soft_thresh(x[i] + step * gradient[i], step * lambda);
        }
    }

    x
}

fn soft_thresh(x: f64, threshold: f64) -> f64 {
    if x.abs() <= threshold {
        0.0
    } else if x > threshold {
        x - threshold
    } else {
        x + threshold
    }
}

/// Generate a random Gaussian sensing matrix
pub fn random_sensing_matrix(m: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng_state = seed;
    let mut matrix = vec![vec![0.0; n]; m];

    for i in 0..m {
        for j in 0..n {
            let (val, new_state) = gaussian_random(rng_state);
            rng_state = new_state;
            matrix[i][j] = val / (m as f64).sqrt();
        }
    }

    matrix
}

/// Generate a sparse signal with given sparsity
pub fn generate_sparse_signal(n: usize, sparsity: usize, seed: u64) -> Vec<f64> {
    let mut signal = vec![0.0; n];
    let mut rng_state = seed;

    let mut used = std::collections::HashSet::new();
    for _ in 0..sparsity {
        let (idx_f, new_state) = uniform_random(rng_state);
        rng_state = new_state;
        let mut idx = (idx_f * n as f64) as usize;
        while used.contains(&idx) {
            idx = (idx + 1) % n;
        }
        used.insert(idx);

        let (val, new_state) = gaussian_random(rng_state);
        rng_state = new_state;
        signal[idx] = val;
    }

    signal
}

/// Compute the coherence of a sensing matrix (max absolute inner product between columns)
pub fn coherence(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix[0].len();
    let m = matrix.len();

    // Extract columns
    let mut cols: Vec<Vec<f64>> = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            cols[j][i] = matrix[i][j];
        }
    }

    let mut max_coherence: f64 = 0.0;
    for i in 0..n {
        let norm_i: f64 = cols[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        for j in (i + 1)..n {
            let dot: f64 = cols[i].iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
            let norm_j: f64 = cols[j].iter().map(|x| x * x).sum::<f64>().sqrt();
            let coherence = (dot / (norm_i * norm_j)).abs();
            max_coherence = max_coherence.max(coherence);
        }
    }

    max_coherence
}

// Helper functions
fn mat_vec_mul(matrix: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum()).collect()
}

fn mat_vec_mul_transpose(matrix: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    let n = matrix[0].len();
    (0..n)
        .map(|j| matrix.iter().zip(vec.iter()).map(|(row, &v)| row[j] * v).sum())
        .collect()
}

fn hard_threshold_keep(x: &mut [f64], k: usize) {
    let mut indexed: Vec<(usize, f64)> = x.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut keep = vec![false; x.len()];
    for (idx, _) in indexed.iter().take(k) {
        keep[*idx] = true;
    }

    for (i, v) in x.iter_mut().enumerate() {
        if !keep[i] {
            *v = 0.0;
        }
    }
}

fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    for col in 0..n {
        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 { continue; }
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-15 {
            x[i] /= aug[i][i];
        }
    }
    x
}

// Simple pseudo-random number generators
fn gaussian_random(state: u64) -> (f64, u64) {
    // Box-Muller transform
    let (u1, s1) = uniform_random(state);
    let (u2, s2) = uniform_random(s1);
    let u1 = u1.max(1e-10);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (z, s2)
}

fn uniform_random(mut state: u64) -> (f64, u64) {
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let x = ((state >> 33) as f64) / (1u64 << 31) as f64;
    (x, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sparse_signal_generation() {
        let signal = generate_sparse_signal(100, 5, 42);
        let nonzero = signal.iter().filter(|&&x| x.abs() > 1e-10).count();
        assert_eq!(nonzero, 5);
    }

    #[test]
    fn test_random_sensing_matrix() {
        let matrix = random_sensing_matrix(20, 100, 42);
        assert_eq!(matrix.len(), 20);
        assert_eq!(matrix[0].len(), 100);
    }

    #[test]
    fn test_iht_recovery() {
        let n = 64;
        let m = 24;
        let sparsity = 4;

        let signal = generate_sparse_signal(n, sparsity, 42);
        let matrix = random_sensing_matrix(m, n, 123);
        let measurements = mat_vec_mul(&matrix, &signal);

        let recovered = iterative_hard_thresholding(&measurements, &matrix, sparsity, 200, 1e-6);

        // Check support recovery
        let mut orig_support: Vec<usize> = signal.iter().enumerate()
            .filter(|(_, &v)| v.abs() > 1e-10).map(|(i, _)| i).collect();
        let mut recovered_support: Vec<usize> = recovered.iter().enumerate()
            .filter(|(_, &v)| v.abs() > 0.1).map(|(i, _)| i).collect();
        orig_support.sort();
        recovered_support.sort();

        // At least some support should be recovered
        let overlap = orig_support.iter().filter(|i| recovered_support.contains(i)).count();
        assert!(overlap >= 2, "Expected at least 2 support items recovered, got {}", overlap);
    }

    #[test]
    fn test_omp_recovery() {
        let n = 32;
        let m = 16;
        let sparsity = 3;

        let mut signal = vec![0.0; n];
        signal[5] = 3.0;
        signal[15] = -2.0;
        signal[25] = 1.5;

        let matrix = random_sensing_matrix(m, n, 77);
        let measurements = mat_vec_mul(&matrix, &signal);

        let result = orthogonal_matching_pursuit(&measurements, &matrix, sparsity);

        assert!(result.support.contains(&5));
        assert!(result.support.contains(&15));
        assert!(result.support.contains(&25));
        assert!(result.residual_norm < 1.0);
    }

    #[test]
    fn test_omp_result_structure() {
        let n = 16;
        let m = 8;
        let matrix = random_sensing_matrix(m, n, 1);
        let measurements = vec![0.0; m];
        let result = orthogonal_matching_pursuit(&measurements, &matrix, 3);
        assert_eq!(result.signal.len(), n);
        assert_eq!(result.support.len(), 3);
    }

    #[test]
    fn test_basis_pursuit_denoising() {
        let n = 32;
        let m = 16;
        let mut signal = vec![0.0; n];
        signal[5] = 3.0;
        signal[15] = -2.0;

        let matrix = random_sensing_matrix(m, n, 99);
        let measurements = mat_vec_mul(&matrix, &signal);

        let recovered = basis_pursuit_denoising(&measurements, &matrix, 0.1, 500);
        assert_eq!(recovered.len(), n);

        // Should recover the sparse structure
        let large_coeffs = recovered.iter().filter(|&&x| x.abs() > 0.5).count();
        assert!(large_coeffs <= 5, "Expected sparse result, got {} large coefficients", large_coeffs);
    }

    #[test]
    fn test_coherence() {
        let matrix = random_sensing_matrix(10, 20, 42);
        let coh = coherence(&matrix);
        assert!(coh >= 0.0 && coh <= 1.0);
    }

    #[test]
    fn test_mat_vec_mul() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vec = vec![1.0, 1.0];
        let result = mat_vec_mul(&matrix, &vec);
        assert_abs_diff_eq!(result[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hard_threshold_keep() {
        let mut x = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        hard_threshold_keep(&mut x, 2);
        assert_eq!(x[3], 0.9); // largest
        assert_eq!(x[1], 0.5); // second largest
        assert_eq!(x[0], 0.0);
    }

    #[test]
    fn test_generate_sparse_signal_different_seeds() {
        let s1 = generate_sparse_signal(50, 3, 1);
        let s2 = generate_sparse_signal(50, 3, 2);
        assert_ne!(s1, s2);
    }
}
