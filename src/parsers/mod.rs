//! Host-side decoders for models whose outputs the device cannot parse
//! itself. Run such models with [`crate::neural_network::NeuralNetworkNode`]
//! and decode the resulting `NNData` here.

pub mod yunet;
