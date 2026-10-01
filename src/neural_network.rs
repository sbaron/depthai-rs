use std::ffi::CString;
use std::time::Duration;

use autocxx::c_int;
use depthai_sys::{depthai, DaiNNData};

use crate::camera::{CameraNode, OutputQueue};
use crate::error::{clear_error_flag, last_error, take_error_if_any, Result};

/// Generic neural network inference node: runs a model on-device and outputs
/// raw tensors (`NNData`), for models whose output the device cannot parse
/// itself (face detectors, embedding models, regressors, ...).
///
/// Ports: input `"in"` (see [`NeuralNetworkNode::input`]), outputs `out`
/// (`NNData`) and `passthrough` (the frame the network ran on).
///
/// Mirrors C++: `dai::node::NeuralNetwork`.
#[crate::native_node_wrapper(native = "dai::node::NeuralNetwork", outputs(out, passthrough))]
pub struct NeuralNetworkNode {
    node: crate::pipeline::Node,
}

impl NeuralNetworkNode {
    /// The network's input port (`"in"`), e.g. to feed it from the host
    /// with [`crate::output::Input::create_input_queue`].
    pub fn input(&self) -> Result<crate::output::Input> {
        self.as_node().input("in")
    }

    /// Feed `camera` into the network and load `model` — a Luxonis HubAI
    /// model slug such as `"yunet:640x360"`, downloaded from the model zoo on
    /// first use (requires network access) and cached. The camera output is
    /// requested at the model's input size and frame type.
    ///
    /// Mirrors C++: `NeuralNetwork::build(camera, model, capability)`.
    pub fn build(&self, camera: &CameraNode, model: &str, config: NeuralNetworkBuildConfig) -> Result<()> {
        clear_error_flag();
        let model_c = CString::new(model).map_err(|_| last_error("model name contains a NUL byte"))?;
        let ok = unsafe {
            depthai::dai_neural_network_build(
                self.node.handle(),
                camera.as_node().handle(),
                model_c.as_ptr(),
                config.fps.unwrap_or(-1.0),
                c_int(config.num_shaves.map(|n| n as i32).unwrap_or(-1)),
                c_int(config.enable_undistortion.map(i32::from).unwrap_or(-1)),
            )
        };
        if ok {
            Ok(())
        } else {
            Err(last_error("failed to build NeuralNetwork"))
        }
    }

    /// Load `model` (HubAI slug, see [`NeuralNetworkNode::build`]) without
    /// linking any input: link one yourself or feed frames from the host.
    pub fn set_model(&self, model: &str, num_shaves: Option<u32>) -> Result<()> {
        clear_error_flag();
        let model_c = CString::new(model).map_err(|_| last_error("model name contains a NUL byte"))?;
        let ok = unsafe {
            depthai::dai_neural_network_set_model(
                self.node.handle(),
                model_c.as_ptr(),
                c_int(num_shaves.map(|n| n as i32).unwrap_or(-1)),
            )
        };
        if ok {
            Ok(())
        } else {
            Err(last_error("failed to load NeuralNetwork model"))
        }
    }

    /// Number of inference threads (RVC2 default: 2). Each thread takes the
    /// model's SHAVE count, so 1 thread halves the SHAVE budget.
    pub fn set_num_inference_threads(&self, num_threads: u32) -> Result<()> {
        clear_error_flag();
        unsafe { depthai::dai_neural_network_set_num_inference_threads(self.node.handle(), c_int(num_threads as i32)) };
        take_error_if_any("failed to set inference threads").map_or(Ok(()), Err)
    }

    /// Number of output message buffers in the pool.
    pub fn set_num_pool_frames(&self, num_frames: u32) -> Result<()> {
        clear_error_flag();
        unsafe { depthai::dai_neural_network_set_num_pool_frames(self.node.handle(), c_int(num_frames as i32)) };
        take_error_if_any("failed to set pool frames").map_or(Ok(()), Err)
    }

    /// Input size `(width, height)` of the loaded model, if known.
    pub fn input_size(&self) -> Option<(u32, u32)> {
        clear_error_flag();
        let (mut w, mut h) = (0u32, 0u32);
        let ok = unsafe { depthai::dai_neural_network_get_input_size(self.node.handle(), &mut w, &mut h) };
        ok.then_some((w, h))
    }
}

/// Options for [`NeuralNetworkNode::build`]; `None` keeps the default.
#[derive(Debug, Clone, Copy, Default)]
pub struct NeuralNetworkBuildConfig {
    /// Camera frame rate requested for the network input.
    pub fps: Option<f32>,
    /// Superblob variant compiled for this many SHAVE cores; the default (8)
    /// may not fit next to StereoDepth or a second network on RVC2.
    pub num_shaves: Option<u32>,
    /// Undistort the network input. Needed when its results are combined
    /// with (always undistorted) aligned depth, e.g. by
    /// `SpatialLocationCalculator`, which rejects mismatched frames.
    pub enable_undistortion: Option<bool>,
}

/// A raw output tensor, dequantized to f32, in row-major order.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

/// Raw inference results of a `NeuralNetworkNode` (`dai::NNData`).
pub struct NNData {
    handle: DaiNNData,
}

unsafe impl Send for NNData {}

impl Drop for NNData {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { depthai::dai_nn_data_release(self.handle) };
        }
    }
}

impl NNData {
    pub(crate) fn handle(&self) -> DaiNNData {
        self.handle
    }

    /// Sequence number of the frame the network ran on; matches the
    /// `passthrough` frame's [`crate::camera::ImageFrame::sequence_num`].
    pub fn sequence_num(&self) -> i64 {
        unsafe { depthai::dai_nn_data_get_sequence_num(self.handle) }
    }

    /// Names of all output layers.
    pub fn layer_names(&self) -> Vec<String> {
        let count: i32 = unsafe { depthai::dai_nn_data_layer_count(self.handle) }.into();
        let mut buf = [0u8; 256];
        (0..count.max(0))
            .filter_map(|i| {
                let ok = unsafe {
                    depthai::dai_nn_data_layer_name(
                        self.handle,
                        c_int(i),
                        buf.as_mut_ptr() as *mut std::os::raw::c_char,
                        c_int(buf.len() as i32),
                    )
                };
                ok.then(|| {
                    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
                    String::from_utf8_lossy(&buf[..len]).into_owned()
                })
            })
            .collect()
    }

    /// Output layer `name`, dequantized to f32.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        clear_error_flag();
        let name_c = CString::new(name).map_err(|_| last_error("layer name contains a NUL byte"))?;
        let mut dims = [0u32; 8];
        let ndims: i32 =
            unsafe { depthai::dai_nn_data_tensor_dims(self.handle, name_c.as_ptr(), dims.as_mut_ptr(), c_int(dims.len() as i32)) }
                .into();
        if ndims < 0 {
            return Err(last_error("no such output layer"));
        }
        let len = unsafe { depthai::dai_nn_data_tensor_f32(self.handle, name_c.as_ptr(), std::ptr::null_mut(), 0) };
        let mut data = vec![0f32; len];
        let copied = unsafe { depthai::dai_nn_data_tensor_f32(self.handle, name_c.as_ptr(), data.as_mut_ptr(), data.len()) };
        if copied != len {
            return Err(last_error("failed to read tensor"));
        }
        Ok(Tensor { dims: dims[..(ndims as usize).min(dims.len())].iter().map(|&d| d as usize).collect(), data })
    }
}

impl OutputQueue {
    /// Pull the next `NNData` message; `None` on timeout.
    pub fn blocking_next_nn_data(&self, timeout: Option<Duration>) -> Result<Option<NNData>> {
        clear_error_flag();
        let timeout_ms = timeout.map(|d| d.as_millis() as i32).unwrap_or(-1);
        let handle = unsafe { depthai::dai_queue_get_nn_data(self.handle(), c_int(timeout_ms)) };
        if handle.is_null() {
            take_error_if_any("failed to pull NNData").map_or(Ok(None), Err)
        } else {
            Ok(Some(NNData { handle }))
        }
    }
}
