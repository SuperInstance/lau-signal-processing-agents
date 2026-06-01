# lau-signal-processing-agents

**Signal processing for agent observations — extract structure from noisy signals**

A Rust library implementing signal processing for agent systems: FFT, FIR/IIR filters, wavelets, Kalman filtering, adaptive filters (LMS/RLS), Wiener filtering, compressed sensing, PSD estimation, windowing, and a composable observation pipeline. Built to turn raw sensor streams into structured features.

111 tests · 12 modules · ~3,200 LOC

---

## What This Does

Agents observe the world through **noisy, high-dimensional signals** — sensor readings, market ticks, audio streams, image pixels. This library provides the signal processing toolkit to extract structure:

- **FFT** — Cooley-Tukey radix-2 FFT, inverse FFT, magnitude/phase spectra, dominant frequency detection
- **Power spectral density** — Periodogram, Welch's method with overlapping segments, spectrogram
- **Windowing** — Hamming, Hanning, Blackman, Kaiser, rectangular, flat-top windows with coherent gain normalization
- **FIR/IIR filters** — Lowpass, highpass, bandpass design; moving average; exponential smoothing; median filter
- **Wavelets** — Haar and Daubechies D4 forward/inverse transform, multi-resolution decomposition, soft/hard thresholding, wavelet denoising
- **Kalman filter** — Linear Kalman with predict/update cycle, constant-position and constant-velocity models, Extended Kalman Filter (EKF)
- **Wiener filter** — Optimal frequency-domain filter from PSD or training signals, Wiener deconvolution
- **Adaptive filters** — LMS, NLMS (normalized LMS), RLS (Recursive Least Squares) with convergence tracking
- **Compressed sensing** — Iterative Hard Thresholding (IHT), Orthogonal Matching Pursuit (OMP), Basis Pursuit Denoising, random sensing matrices, coherence analysis
- **Pipeline** — Composable stages: window → filter → normalize → downsample → extract features

---

## Key Idea

Classical signal processing assumes you know the signal model. Agents don't — they discover it. This library combines:

1. **Standard DSP** (FFT, filters, wavelets) for the known parts
2. **Adaptive methods** (LMS, RLS, Kalman) that learn parameters online
3. **Sparse recovery** (compressed sensing) for underdetermined observations
4. **A pipeline abstraction** that chains processing stages into a feature extractor

The result: raw observations → structured features, with each stage tunable or learned.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-signal-processing-agents = { git = "https://github.com/SuperInstance/lau-signal-processing-agents" }
```

### Dependencies

- `nalgebra` 0.33 — linear algebra (Kalman filter)
- `num-complex` 0.4 — complex number arithmetic
- `num-traits` 0.2 — numeric traits
- `serde` 1.x (with `derive`) — serialization

---

## Quick Start

```rust
use lau_signal_processing_agents::*;

// FFT: find the dominant frequency in a signal
let signal = /* 256 samples at 44100 Hz */;
let spectrum = fft(&signal);
let freq = dominant_frequency(&signal, 44100.0);

// Filter out high-frequency noise
let mut lpf = filtering::FirFilter::lowpass(1000.0, 44100.0, 51);
let filtered = lpf.filter(&signal);

// Wavelet denoising
let denoised = wavelet::wavelet_denoise(&signal, 4);

// Kalman filter: track a moving target
let mut kf = kalman::KalmanFilter::constant_velocity_1d(0.1, 1.0);
let estimates = kf.filter_sequence(&measurements);

// Power spectral density via Welch's method
let welch_result = psd::welch(&signal, 44100.0, 64, 0.5, psd::WindowType::Hanning);
println!("Peak freq: {} Hz", psd::peak_frequency(&welch_result.frequencies, &welch_result.psd));

// Adaptive filtering (LMS)
let mut lms = adaptive::LmsFilter::new(32, 0.01);
let result = lms.filter_sequence(&inputs, &desired_outputs);
println!("Final MSE: {}", result.final_mse);

// Compressed sensing: recover a sparse signal from 30% measurements
let sensing = compressed_sensing::random_sensing_matrix(m, n, 42);
let recovered = compressed_sensing::orthogonal_matching_pursuit(
    &measurements, &sensing, sparsity, 100, 1e-6
);

// Pipeline: compose stages
let mut pipe = pipeline::Pipeline::new();
pipe.add_stage(pipeline::PipelineStage::FirFilter { coeffs: lpf.coeffs.clone() })
    .add_stage(pipeline::PipelineStage::Normalize)
    .add_stage(pipeline::PipelineStage::MagnitudeSpectrum);
let result = pipe.run(&signal);
```

---

## API Reference

### `fourier` — Fourier Transform

| Function | Description |
|---|---|
| `dft(signal)` | Naive DFT, O(n²) |
| `idft(spectrum)` | Inverse DFT |
| `fft(signal)` | Cooley-Tukey radix-2 FFT, O(n log n) |
| `ifft(spectrum)` | Inverse FFT |
| `fft_complex(input)` | Complex-to-complex FFT |
| `zero_pad_to_power2(signal)` | Pad signal to next power of 2 |
| `frequency_bins(n, fs)` | Frequency axis for FFT output |
| `dominant_frequency(signal, fs)` | Find the strongest frequency component |
| `magnitude_spectrum(signal)` | Magnitude of FFT |
| `phase_spectrum(signal)` | Phase of FFT |

### `filtering` — FIR and IIR Filters

| Type / Function | Description |
|---|---|
| `FirFilter` | Finite impulse response filter |
| `FirFilter::moving_average(order)` | Uniform moving average |
| `FirFilter::lowpass(cutoff, fs, order)` | Windowed-sinc lowpass |
| `FirFilter::highpass(cutoff, fs, order)` | Highpass via spectral inversion |
| `FirFilter::bandpass(lo, hi, fs, order)` | Bandpass filter |
| `FirFilter::filter(&self, signal)` | Apply filter (convolution) |
| `IirFilter` | Infinite impulse response filter (biquad) |
| `IirFilter::lowpass_1st(cutoff, fs)` | First-order IIR lowpass |
| `IirFilter::highpass_1st(cutoff, fs)` | First-order IIR highpass |
| `exponential_smoothing(signal, α)` | Exponential moving average |
| `median_filter(signal, window)` | Nonlinear median filter |
| `differentiate(signal)` | Numerical differentiation |
| `integrate(signal)` | Numerical integration |

### `wavelet` — Wavelet Transform

| Type / Function | Description |
|---|---|
| `WaveletType` | `Haar`, `Daubechies4` |
| `WaveletDecomposition` | Multi-resolution decomposition result |
| `haar_forward(signal)` | Single-level Haar forward transform |
| `haar_decompose(signal, levels)` | Multi-level Haar decomposition |
| `haar_reconstruct(decomp)` | Perfect reconstruction from Haar decomposition |
| `db4_forward(signal)` | Single-level Daubechies D4 forward |
| `db4_decompose(signal, levels)` | Multi-level Daubechies D4 decomposition |
| `soft_threshold(coeffs, λ)` | Soft thresholding (wavelet denoising) |
| `hard_threshold(coeffs, λ)` | Hard thresholding |
| `universal_threshold(signal)` | VisuShrink universal threshold σ√(2 log n) |
| `wavelet_denoise(signal, levels)` | Denoise via wavelet thresholding |

### `kalman` — Kalman Filter

| Type / Function | Description |
|---|---|
| `KalmanFilter` | Linear Kalman filter |
| `KalmanFilter::constant_position_1d(q, r)` | 1D constant-position model |
| `KalmanFilter::constant_velocity_1d(q, r)` | 1D constant-velocity model |
| `kf.predict()` | Prediction step (time update) |
| `kf.update(measurement)` | Update step (measurement update) |
| `kf.step(measurement)` | Predict + update combined |
| `kf.filter_sequence(measurements)` | Filter a whole sequence |
| `kf.state_estimate()` | Current state estimate |
| `kf.uncertainty()` | Current uncertainty (trace of covariance) |
| `ExtendedKalmanFilter` | EKF for nonlinear systems |

### `adaptive` — Adaptive Filters

| Type / Function | Description |
|---|---|
| `LmsFilter` | Least Mean Squares adaptive filter |
| `LmsFilter::new(order, μ)` | Create with given order and step size |
| `LmsFilter::process(input, desired)` | Process one sample, return (output, error) |
| `LmsFilter::filter_sequence(inputs, desired)` | Process whole sequence |
| `NlmsFilter` | Normalized LMS (step size adapts to input power) |
| `RlsFilter` | Recursive Least Squares (fast convergence) |
| `RlsFilter::new(order, λ, δ)` | Create with forgetting factor and regularization |
| `LmsResult` | Result: filtered output, error sequence, final MSE, weight history |

### `wiener` — Wiener Filter

| Type / Function | Description |
|---|---|
| `WienerFilter` | Optimal frequency-domain filter |
| `WienerFilter::from_psd(signal_psd, noise_psd)` | Design from known PSDs |
| `WienerFilter::from_signals(clean, noisy)` | Estimate PSDs from training data |
| `wf.filter(noisy_signal)` | Apply Wiener filter |
| `WienerDeconvolution` | Wiener deconvolution for system identification |
| `wiener_fir(desired, input, order)` | Time-domain FIR Wiener filter |

### `compressed_sensing` — Sparse Recovery

| Function | Description |
|---|---|
| `iterative_hard_thresholding(measurements, A, k, max_iter, tol)` | IHT recovery |
| `orthogonal_matching_pursuit(measurements, A, k, max_iter, tol)` | OMP recovery |
| `basis_pursuit_denoising(measurements, A, λ, max_iter, tol)` | BPDN via ISTA |
| `random_sensing_matrix(m, n, seed)` | Gaussian random sensing matrix |
| `generate_sparse_signal(n, k, seed)` | Generate test sparse signal |
| `coherence(matrix)` | Mutual coherence μ(A) |
| `OmpResult` | OMP result with recovered signal and support |

### `psd` — Power Spectral Density

| Function | Description |
|---|---|
| `periodogram(signal, fs)` | Basic PSD estimate |
| `welch(signal, fs, segment_len, overlap, window)` | Welch's method (averaged, windowed) |
| `total_power(psd, df)` | Parseval's theorem: total signal power |
| `peak_frequency(freqs, psd)` | Frequency of strongest spectral component |
| `spectrogram(signal, fs, seg_len, overlap, window)` | Time-frequency spectrogram |
| `WelchResult` | PSD, frequency axis, number of segments |
| `SpectrogramResult` | 2D PSD array with time/frequency axes |

### `windowing` — Window Functions

| Function | Description |
|---|---|
| `hamming(n)` | Hamming window |
| `hanning(n)` | Hanning (Hann) window |
| `blackman(n)` | Blackman window |
| `kaiser(n, β)` | Kaiser window with adjustable sidelobe attenuation |
| `rectangular(n)` | Rectangular (Dirichlet) window |
| `flat_top(n)` | Flat-top window for amplitude accuracy |
| `apply_window(signal, window)` | Element-wise multiplication |
| `coherent_gain(window)` | Window coherent gain (mean value) |
| `normalize_window(window)` | Normalize to unit coherent gain |

### `pipeline` — Observation Pipeline

| Type / Function | Description |
|---|---|
| `Pipeline` | Composable processing chain |
| `PipelineStage` | `Window`, `FirFilter`, `Normalize`, `Downsample`, `MagnitudeSpectrum`, `DominantFrequency`, `Custom` |
| `Pipeline::new()` | Create empty pipeline |
| `Pipeline::add_stage(stage)` | Append a stage (builder pattern) |
| `Pipeline::run(input)` | Execute all stages |
| `PipelineResult` | Output signal, intermediate results, metadata |

---

## How It Works

### Architecture

```
windowing (window functions)
    └── fourier (DFT, FFT)
        ├── psd (periodogram, Welch, spectrogram)
        └── wiener (frequency-domain optimal filter)
filtering (FIR, IIR, smoothing)
wavelet (Haar, Daubechies D4, denoising)
kalman (linear Kalman, EKF)
adaptive (LMS, NLMS, RLS)
compressed_sensing (IHT, OMP, BPDN)
pipeline (composable stages)
```

### FFT Implementation

The FFT uses the **Cooley-Tukey radix-2** decimation-in-time algorithm:

1. Pad signal to power of 2 (or use `zero_pad_to_power2`)
2. Bit-reverse the input order
3. Butterfly operations at each stage: O(n log n)

The inverse FFT conjugates, forward-transforms, conjugates again, and divides by n.

### Wavelet Denoising Pipeline

1. **Decompose** signal into approximation (low-freq) and detail (high-freq) coefficients at multiple levels
2. **Threshold** the detail coefficients using universal threshold λ = σ√(2 log n)
3. **Reconstruct** from thresholded coefficients

This preserves low-frequency structure while removing high-frequency noise — superior to simple lowpass filtering for transient signals.

### Kalman Filter Cycle

Each time step:

1. **Predict**: `x̂ₖ|ₖ₋₁ = F·x̂ₖ₋₁`, `Pₖ|ₖ₋₁ = F·Pₖ₋₁·Fᵀ + Q`
2. **Update**: `Kₖ = Pₖ|ₖ₋₁·Hᵀ·(H·Pₖ|ₖ₋₁·Hᵀ + R)⁻¹`, then correct state and covariance

The Kalman gain K optimally trades off prediction vs. measurement based on their uncertainties.

### Compressed Sensing

Recover a sparse signal x ∈ ℝⁿ from m << n measurements y = Ax using:

- **IHT**: Iterative gradient descent + hard thresholding to sparsity k
- **OMP**: Greedy selection of atoms from the sensing matrix, orthogonal projection at each step
- **BPDN**: L1-regularized minimization via ISTA (Iterative Shrinkage-Thresholding Algorithm)

Recovery is guaranteed when the sensing matrix has low coherence μ(A) and m = O(k · log(n/k)).

---

## The Math

### Discrete Fourier Transform

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi kn / N}$$

The FFT computes this in O(N log N) via the Cooley-Tukey butterfly factorization.

### Wiener Filter

The frequency-domain Wiener filter minimizes mean-square error:

$$H(f) = \frac{S_{xx}(f)}{S_{xx}(f) + S_{nn}(f)}$$

where S_xx is the signal PSD and S_nn is the noise PSD. At frequencies where noise dominates, H(f) → 0; where signal dominates, H(f) → 1.

### Wavelet Transform

The discrete wavelet transform decomposes a signal into approximation coefficients aⱼ and detail coefficients dⱼ at each level j:

$$a_{j+1}[n] = \sum_k h[k] \cdot a_j[2n - k] \quad \text{(lowpass)}$$
$$d_{j+1}[n] = \sum_k g[k] \cdot a_j[2n - k] \quad \text{(highpass)}$$

For Haar: h = [1/√2, 1/√2], g = [1/√2, -1/√2]. For Daubechies D4: 4-tap filters with better regularity.

### Kalman Filter

The optimal linear estimator for the state-space model:

$$x_{k+1} = F x_k + w_k, \quad w_k \sim \mathcal{N}(0, Q)$$
$$y_k = H x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)$$

The Kalman filter is the Bayes-optimal sequential estimator for this linear-Gaussian system.

### Compressed Sensing

Given y = Ax where A ∈ ℝᵐˣⁿ with m << n, recover the k-sparse x via:

$$\min \|x\|_1 \quad \text{subject to} \quad \|Ax - y\|_2 \le \epsilon$$

Recovery is exact when A satisfies the Restricted Isometry Property (RIP) of order 2k.

---

## Running Tests

```bash
cargo test
```

111 tests: FFT correctness, filter frequency response, wavelet reconstruction, Kalman tracking, adaptive convergence, compressed sensing recovery, PSD estimation, windowing properties, and pipeline execution.

---

## License

MIT
