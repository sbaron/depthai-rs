use std::ffi::CString;
use std::time::Duration;

use autocxx::c_int;
use depthai_sys::{depthai, DaiSpatialDetections};

use crate::camera::{CameraNode, OutputQueue};
use crate::error::{clear_error_flag, last_error, take_error_if_any, Result};
use crate::stereo_depth::StereoDepthNode;

/// A single spatial detection: 2D bounding box in normalized [0,1] image
/// coordinates plus the estimated 3D position in millimeters (camera frame).
#[derive(Debug, Clone)]
pub struct SpatialDetection {
    pub label: u32,
    pub label_name: String,
    pub confidence: f32,
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub x_mm: f32,
    pub y_mm: f32,
    pub z_mm: f32,
    /// Keypoints (e.g. face landmarks), when the model provides them.
    pub keypoints: Vec<SpatialKeypoint>,
}

/// A detection keypoint: normalized [0,1] image position plus its 3D
/// position in millimeters (0 when no depth was available).
#[derive(Debug, Clone, Copy)]
pub struct SpatialKeypoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub x_mm: f32,
    pub y_mm: f32,
    pub z_mm: f32,
}

// No `inputs(..)`/`outputs(..)` on the macro: this node is a C++ node group
// whose ports alias subnode ports, which the generic by-name lookup cannot
// resolve. Outputs are exposed through `output()` below instead.
#[crate::native_node_wrapper(native = "dai::node::SpatialDetectionNetwork")]
pub struct SpatialDetectionNetworkNode {
    node: crate::pipeline::Node,
}

impl SpatialDetectionNetworkNode {
    /// Resolve one of the node's outputs: `"out"` (SpatialImgDetections),
    /// `"outNetwork"`, `"passthrough"` or `"passthroughDepth"`.
    pub fn output(&self, name: &str) -> Result<crate::output::Output> {
        clear_error_flag();
        let name_c = CString::new(name).map_err(|_| last_error("invalid output name"))?;
        let handle = unsafe {
            depthai::dai_spatial_detection_network_get_output(self.node.handle(), name_c.as_ptr())
        };
        if handle.is_null() {
            Err(last_error("failed to get SpatialDetectionNetwork output"))
        } else {
            Ok(crate::output::Output::from_handle(
                std::sync::Arc::clone(&self.node.pipeline),
                handle,
            ))
        }
    }

    /// The parsed detections output (`SpatialImgDetections` messages).
    pub fn out(&self) -> Result<crate::output::Output> {
        self.output("out")
    }

    /// Connect `camera` and `stereo` into the network and load `model` — a
    /// Luxonis HubAI model slug (e.g. `"yolov6-nano"`), downloaded from the
    /// model zoo on first use (requires network access) and cached.
    ///
    /// `num_shaves` limits the SHAVE cores the model is compiled for; on RVC2
    /// the superblob default (8) may not fit next to StereoDepth (7 free).
    ///
    /// Mirrors C++: `SpatialDetectionNetwork::build(camera, stereo, model, fps)`.
    pub fn build(
        &self,
        camera: &CameraNode,
        stereo: &StereoDepthNode,
        model: &str,
        fps: Option<f32>,
        num_shaves: Option<u32>,
    ) -> Result<()> {
        clear_error_flag();
        let model_c =
            CString::new(model).map_err(|_| last_error("model name contains a NUL byte"))?;
        let ok = unsafe {
            depthai::dai_spatial_detection_network_build(
                self.node.handle(),
                camera.as_node().handle(),
                stereo.as_node().handle(),
                model_c.as_ptr(),
                fps.unwrap_or(-1.0),
                c_int(num_shaves.map(|n| n as i32).unwrap_or(-1)),
            )
        };
        if ok {
            Ok(())
        } else {
            Err(last_error("failed to build SpatialDetectionNetwork"))
        }
    }

    /// Discard detections below this confidence (0..1).
    pub fn set_confidence_threshold(&self, threshold: f32) {
        clear_error_flag();
        unsafe {
            depthai::dai_spatial_detection_network_set_confidence_threshold(
                self.node.handle(),
                threshold,
            )
        };
    }

    /// Scale factor applied to the bounding box before sampling depth for the
    /// spatial coordinates (e.g. 0.5 = use the central half of the box).
    pub fn set_bounding_box_scale_factor(&self, factor: f32) {
        clear_error_flag();
        unsafe {
            depthai::dai_spatial_detection_network_set_bounding_box_scale_factor(
                self.node.handle(),
                factor,
            )
        };
    }

    /// Depth values outside [lower_mm, upper_mm] are ignored when computing
    /// the spatial coordinates.
    pub fn set_depth_thresholds(&self, lower_mm: u32, upper_mm: u32) {
        clear_error_flag();
        unsafe {
            depthai::dai_spatial_detection_network_set_depth_thresholds(
                self.node.handle(),
                lower_mm,
                upper_mm,
            )
        };
    }

    /// Class label map of the loaded model, if the model provides one.
    pub fn classes(&self) -> Option<Vec<String>> {
        clear_error_flag();
        let count =
            unsafe { depthai::dai_spatial_detection_network_class_count(self.node.handle()) };
        let count: i32 = count.into();
        if count < 0 {
            return None;
        }
        let mut names = Vec::with_capacity(count as usize);
        let mut buf = [0u8; 256];
        for i in 0..count {
            let ok = unsafe {
                depthai::dai_spatial_detection_network_class_name(
                    self.node.handle(),
                    c_int(i),
                    buf.as_mut_ptr() as *mut std::os::raw::c_char,
                    c_int(buf.len() as i32),
                )
            };
            if !ok {
                return None;
            }
            let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            names.push(String::from_utf8_lossy(&buf[..len]).into_owned());
        }
        Some(names)
    }
}

/// One batch of detections pulled from the network's `out` queue.
pub struct SpatialDetections {
    handle: DaiSpatialDetections,
}

impl Drop for SpatialDetections {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { depthai::dai_spatial_detections_release(self.handle) };
        }
    }
}

impl SpatialDetections {
    pub(crate) fn from_handle(handle: DaiSpatialDetections) -> Self {
        Self { handle }
    }

    pub fn len(&self) -> usize {
        let count: i32 = unsafe { depthai::dai_spatial_detections_count(self.handle) }.into();
        count.max(0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<SpatialDetection> {
        let mut det = SpatialDetection {
            label: 0,
            label_name: String::new(),
            confidence: 0.0,
            xmin: 0.0,
            ymin: 0.0,
            xmax: 0.0,
            ymax: 0.0,
            x_mm: 0.0,
            y_mm: 0.0,
            z_mm: 0.0,
            keypoints: Vec::new(),
        };
        let ok = unsafe {
            depthai::dai_spatial_detections_get(
                self.handle,
                c_int(index as i32),
                &mut det.label,
                &mut det.confidence,
                &mut det.xmin,
                &mut det.ymin,
                &mut det.xmax,
                &mut det.ymax,
                &mut det.x_mm,
                &mut det.y_mm,
                &mut det.z_mm,
            )
        };
        if !ok {
            return None;
        }
        let mut buf = [0u8; 256];
        let ok = unsafe {
            depthai::dai_spatial_detections_label_name(
                self.handle,
                c_int(index as i32),
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                c_int(buf.len() as i32),
            )
        };
        if ok {
            let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            det.label_name = String::from_utf8_lossy(&buf[..len]).into_owned();
        }
        let count: i32 = unsafe { depthai::dai_spatial_detections_keypoint_count(self.handle, c_int(index as i32)) }.into();
        for k in 0..count.max(0) {
            let mut kp = SpatialKeypoint { x: 0.0, y: 0.0, confidence: 0.0, x_mm: 0.0, y_mm: 0.0, z_mm: 0.0 };
            let ok = unsafe {
                depthai::dai_spatial_detections_get_keypoint(
                    self.handle,
                    c_int(index as i32),
                    c_int(k),
                    &mut kp.x,
                    &mut kp.y,
                    &mut kp.confidence,
                    &mut kp.x_mm,
                    &mut kp.y_mm,
                    &mut kp.z_mm,
                )
            };
            if ok {
                det.keypoints.push(kp);
            }
        }
        Some(det)
    }

    /// Sequence number of the frame the detections were computed on.
    pub fn sequence_num(&self) -> i64 {
        unsafe { depthai::dai_spatial_detections_get_sequence_num(self.handle) }
    }

    pub fn to_vec(&self) -> Vec<SpatialDetection> {
        (0..self.len()).filter_map(|i| self.get(i)).collect()
    }
}

impl OutputQueue {
    /// Pull the next `SpatialImgDetections` message; `None` on timeout.
    pub fn blocking_next_spatial_detections(
        &self,
        timeout: Option<Duration>,
    ) -> Result<Option<SpatialDetections>> {
        clear_error_flag();
        let timeout_ms = timeout.map(|d| d.as_millis() as i32).unwrap_or(-1);
        let msg =
            unsafe { depthai::dai_queue_get_spatial_detections(self.handle(), c_int(timeout_ms)) };
        if msg.is_null() {
            if let Some(err) = take_error_if_any("failed to pull spatial detections") {
                Err(err)
            } else {
                Ok(None)
            }
        } else {
            Ok(Some(SpatialDetections::from_handle(msg)))
        }
    }
}
