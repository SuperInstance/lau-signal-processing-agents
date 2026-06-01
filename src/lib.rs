//! # lau-signal-processing-agents
//!
//! Signal processing for agent observations — agents observe the world through
//! noisy signals and need to extract structure.
//!
//! ## Modules
//! - `fourier`: DFT, FFT (Cooley-Tukey radix-2), frequency analysis
//! - `psd`: Power spectral density (Welch's method, periodogram)
//! - `windowing`: Hamming, Hanning, Blackman, Kaiser windows
//! - `filtering`: FIR and IIR filters
//! - `wavelet`: Haar, Daubechies D4, multi-resolution analysis
//! - `kalman`: Kalman filter for linear Gaussian systems
//! - `wiener`: Wiener optimal linear filter
//! - `adaptive`: LMS and RLS adaptive filters
//! - `compressed_sensing`: Recover sparse signals from few observations
//! - `pipeline`: Agent observation pipeline — raw signal → filtered → features

#![allow(clippy::needless_range_loop, clippy::get_first, clippy::manual_is_multiple_of, clippy::manual_abs_diff)]

pub mod adaptive;
pub mod compressed_sensing;
pub mod filtering;
pub mod fourier;
pub mod kalman;
pub mod pipeline;
pub mod psd;
pub mod wavelet;
pub mod wiener;
pub mod windowing;

pub use num_complex::Complex64;
pub use fourier::{fft, ifft, zero_pad_to_power2};
