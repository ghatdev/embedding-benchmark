//! Embedding backends abstraction
//!
//! Provides a unified trait for different embedding implementations:
//! - fastembed (ONNX runtime)
//! - mistral.rs (Candle + Metal)

pub mod fastembed_backend;
pub mod mistralrs_backend;
pub mod traits;

pub use fastembed_backend::{FastEmbedBackend, FastEmbedModel};
pub use mistralrs_backend::{MistralRsBackend, MistralRsModel, MistralRsQuantization, MistralRsDevice};
pub use traits::{EmbedderBackend, ResourceMetrics};
