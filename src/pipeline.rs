//! Agent observation pipeline — raw signal → filtered → features
//!
//! Provides a pipeline abstraction for transforming raw agent observations
//! into structured features through a sequence of signal processing stages.

use crate::filtering;
use crate::fourier;
use crate::windowing;

/// A single stage in the signal processing pipeline
#[derive(Debug, Clone)]
pub enum PipelineStage {
    /// Apply a window function before processing
    Window(String),
    /// FIR filtering with given coefficients
    FirFilter { coeffs: Vec<f64> },
    /// Normalize the signal to zero mean, unit variance
    Normalize,
    /// Downsample by a factor
    Downsample { factor: usize },
    /// Extract magnitude spectrum (FFT-based)
    MagnitudeSpectrum,
    /// Extract dominant frequency
    DominantFrequency { sample_rate: f64 },
    /// Custom transformation (closure-like, represented as index into user-provided list)
    Custom(usize),
}

/// Result of running a pipeline
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The processed signal after all stages
    pub signal: Vec<f64>,
    /// Extracted features as key-value pairs
    pub features: Vec<(String, f64)>,
}

/// Signal processing pipeline for agent observations
#[derive(Debug, Clone)]
pub struct Pipeline {
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// Create a new empty pipeline
    pub fn new() -> Self {
        Pipeline { stages: Vec::new() }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(&mut self, stage: PipelineStage) -> &mut Self {
        self.stages.push(stage);
        self
    }

    /// Run the pipeline on an input signal
    pub fn run(&self, input: &[f64]) -> PipelineResult {
        let mut signal = input.to_vec();
        let mut features = Vec::new();

        for stage in &self.stages {
            match stage {
                PipelineStage::Window(name) => {
                    let window = match name.as_str() {
                        "hamming" => windowing::hamming(signal.len()),
                        "hanning" => windowing::hanning(signal.len()),
                        "blackman" => windowing::blackman(signal.len()),
                        _ => windowing::hamming(signal.len()),
                    };
                    signal = signal.iter().zip(window.iter()).map(|(&s, &w)| s * w).collect();
                }
                PipelineStage::FirFilter { coeffs } => {
                    let fir = filtering::FirFilter::new(coeffs.clone());
                    signal = fir.filter(&signal);
                }
                PipelineStage::Normalize => {
                    let mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
                    let std_dev: f64 =
                        (signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64).sqrt();
                    if std_dev > 1e-15 {
                        signal = signal.iter().map(|x| (x - mean) / std_dev).collect();
                    }
                }
                PipelineStage::Downsample { factor } => {
                    signal = signal.iter().enumerate()
                        .filter(|(i, _)| i % factor == 0)
                        .map(|(_, &v)| v)
                        .collect();
                }
                PipelineStage::MagnitudeSpectrum => {
                    let mag = fourier::magnitude_spectrum(&signal);
                    features.push(("spectral_energy".to_string(), mag.iter().sum()));
                    features.push(("spectral_peak".to_string(), mag.iter().cloned().fold(0.0_f64, f64::max)));
                }
                PipelineStage::DominantFrequency { sample_rate } => {
                    let freq = fourier::dominant_frequency(&signal, *sample_rate);
                    features.push(("dominant_frequency".to_string(), freq));
                }
                PipelineStage::Custom(_idx) => {
                    // No-op placeholder; users extend via their own logic
                }
            }
        }

        PipelineResult { signal, features }
    }

    /// Number of stages in the pipeline
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether the pipeline has no stages
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_pipeline() {
        let p = Pipeline::new();
        let input = vec![1.0, 2.0, 3.0];
        let result = p.run(&input);
        assert_eq!(result.signal, input);
    }

    #[test]
    fn test_normalize_stage() {
        let mut p = Pipeline::new();
        p.add_stage(PipelineStage::Normalize);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = p.run(&input);
        let mean: f64 = result.signal.iter().sum::<f64>() / result.signal.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_len() {
        let mut p = Pipeline::new();
        assert!(p.is_empty());
        p.add_stage(PipelineStage::Normalize);
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn test_window_stage() {
        let mut p = Pipeline::new();
        p.add_stage(PipelineStage::Window("hamming".to_string()));
        let input = vec![1.0; 64];
        let result = p.run(&input);
        // Hamming window is not all ones, so values should differ
        assert!(result.signal.iter().any(|&v| v != 1.0));
    }

    #[test]
    fn test_downsample_stage() {
        let mut p = Pipeline::new();
        p.add_stage(PipelineStage::Downsample { factor: 2 });
        let input: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let result = p.run(&input);
        assert_eq!(result.signal.len(), 5);
        assert_eq!(result.signal[0], 0.0);
        assert_eq!(result.signal[1], 2.0);
    }

    #[test]
    fn test_magnitude_spectrum_stage() {
        let mut p = Pipeline::new();
        p.add_stage(PipelineStage::MagnitudeSpectrum);
        let input = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let result = p.run(&input);
        assert!(!result.features.is_empty());
    }

    #[test]
    fn test_multi_stage_pipeline() {
        let mut p = Pipeline::new();
        p.add_stage(PipelineStage::Window("hanning".to_string()));
        p.add_stage(PipelineStage::Normalize);
        p.add_stage(PipelineStage::MagnitudeSpectrum);
        let input: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = p.run(&input);
        assert!(!result.features.is_empty());
    }
}
