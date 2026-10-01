use autocxx::c_int;
use depthai_sys::depthai;

use crate::error::{clear_error_flag, last_error, take_error_if_any, Result};
use crate::host_node::Buffer;
use crate::neural_network::NNData;

/// How a ROI's depth samples are reduced to one distance.
///
/// Mirrors C++: `dai::SpatialLocationCalculatorAlgorithm`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialLocationCalculatorAlgorithm {
    Average = 0,
    Min = 1,
    Max = 2,
    Mode = 3,
    Median = 4,
}

/// Computes 3D positions on-device from a depth stream: for detections fed
/// to `inputDetections` (bounding boxes and keypoints), or for configured
/// ROIs.
///
/// Typical use with a host-side parser: link aligned depth to `inputDepth`,
/// send [`ImgDetections`] from the host to `inputDetections`, and read
/// `SpatialImgDetections` from `outputDetections` with
/// [`crate::camera::OutputQueue::blocking_next_spatial_detections`].
///
/// Mirrors C++: `dai::node::SpatialLocationCalculator`.
#[crate::native_node_wrapper(
    native = "dai::node::SpatialLocationCalculator",
    inputs(inputConfig, inputDetections, inputDepth),
    outputs(out, outputDetections, passthroughDepth)
)]
pub struct SpatialLocationCalculatorNode {
    node: crate::pipeline::Node,
}

impl SpatialLocationCalculatorNode {
    /// Depth values outside [lower_mm, upper_mm] are ignored.
    pub fn set_depth_thresholds(&self, lower_mm: u32, upper_mm: u32) {
        clear_error_flag();
        unsafe { depthai::dai_spatial_location_calculator_set_depth_thresholds(self.node.handle(), lower_mm, upper_mm) };
    }

    pub fn set_calculation_algorithm(&self, algorithm: SpatialLocationCalculatorAlgorithm) {
        clear_error_flag();
        unsafe {
            depthai::dai_spatial_location_calculator_set_calculation_algorithm(self.node.handle(), c_int(algorithm as i32))
        };
    }

    /// Scale factor applied to detection boxes before sampling depth
    /// (e.g. 0.5 = use the central half of the box).
    pub fn set_bounding_box_scale_factor(&self, factor: f32) {
        clear_error_flag();
        unsafe { depthai::dai_spatial_location_calculator_set_bounding_box_scale_factor(self.node.handle(), factor) };
    }

    /// Radius in pixels of the depth window sampled around each keypoint.
    pub fn set_keypoint_radius(&self, radius: u32) {
        clear_error_flag();
        unsafe { depthai::dai_spatial_location_calculator_set_keypoint_radius(self.node.handle(), c_int(radius as i32)) };
    }

    /// Also compute a 3D position for each detection keypoint (default: on).
    pub fn set_calculate_spatial_keypoints(&self, enable: bool) {
        clear_error_flag();
        unsafe { depthai::dai_spatial_location_calculator_set_calculate_spatial_keypoints(self.node.handle(), enable) };
    }
}

/// Detections built on the host (e.g. by [`crate::parsers::yunet`]), to be
/// sent to a device node such as `SpatialLocationCalculator.inputDetections`
/// with [`crate::queue::InputQueue::send_buffer`] on [`ImgDetections::as_buffer`].
///
/// Mirrors C++: `dai::ImgDetections`.
pub struct ImgDetections {
    buffer: Buffer,
}

impl ImgDetections {
    /// Empty detections for the frame `source` was computed on: sequence
    /// number, timestamps and image transformation are copied, so the device
    /// can map them onto other frames (e.g. aligned depth).
    pub fn for_nn_data(source: &NNData) -> Result<Self> {
        clear_error_flag();
        let handle = unsafe { depthai::dai_img_detections_new_from_nn_data(source.handle()) };
        if handle.is_null() {
            Err(last_error("failed to create ImgDetections"))
        } else {
            Ok(Self { buffer: Buffer::from_handle(handle) })
        }
    }

    /// Add a detection. Box `[xmin, ymin, xmax, ymax]` and keypoints are
    /// normalized [0,1] image coordinates.
    pub fn push(&mut self, label: u32, confidence: f32, bbox: [f32; 4], keypoints: &[(f32, f32)]) -> Result<()> {
        clear_error_flag();
        let flat: Vec<f32> = keypoints.iter().flat_map(|&(x, y)| [x, y]).collect();
        let ok = unsafe {
            depthai::dai_img_detections_add(
                self.buffer.handle(),
                label,
                confidence,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                if flat.is_empty() { std::ptr::null() } else { flat.as_ptr() },
                c_int(keypoints.len() as i32),
            )
        };
        if ok {
            Ok(())
        } else {
            Err(take_error_if_any("failed to add detection").unwrap_or_else(|| last_error("failed to add detection")))
        }
    }

    pub fn as_buffer(&self) -> &Buffer {
        &self.buffer
    }
}
