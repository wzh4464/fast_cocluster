pub mod normalization;
pub mod tri_factor_base;
pub mod update_rules;

pub mod nbvd;

#[cfg(test)]
pub(crate) mod test_utils;

// Re-exports for convenience
pub use nbvd::NbvdClusterer;
pub use tri_factor_base::{TriFactorConfig, TriFactorResult, TriFactorUpdater};
