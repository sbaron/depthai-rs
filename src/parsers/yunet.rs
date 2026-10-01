//! Decoder for the YuNet face detector (Luxonis model zoo `yunet:*`).
//!
//! The device cannot parse YuNet's output (its `YuNetParser` head is not one
//! of the on-device YOLO/SSD parsers), so run the model with
//! [`crate::neural_network::NeuralNetworkNode`] and decode its `NNData` here.
//! The decoding follows OpenCV's `FaceDetectorYN`: prior boxes over four
//! feature maps (strides 8/16/32/64), `loc`/`conf`/`iou` outputs, score
//! `sqrt(cls * iou)`, then NMS.
//!
//! ```no_run
//! # use depthai::parsers::yunet::YuNetParser;
//! # fn f(nn: &depthai::neural_network::NNData) -> depthai::Result<()> {
//! let parser = YuNetParser::new(640, 360);
//! for face in parser.parse(nn)? {
//!     println!("face {:.0}% at {:?}, left eye {:?}", face.confidence * 100.0, face.bbox, face.landmarks[1]);
//! }
//! # Ok(()) }
//! ```

use crate::error::Result;
use crate::neural_network::NNData;
use crate::spatial_location_calculator::ImgDetections;

const MIN_SIZES: [&[f32]; 4] = [&[10.0, 16.0, 24.0], &[32.0, 48.0], &[64.0, 96.0], &[128.0, 192.0, 256.0]];
const STEPS: [f32; 4] = [8.0, 16.0, 32.0, 64.0];
const VARIANCE: [f32; 2] = [0.1, 0.2];
/// Values per prior in the `loc` output: box (4) + 5 landmarks (10).
const LOC_STRIDE: usize = 14;

/// Landmark order in [`FaceDetection::landmarks`] (as seen by the subject:
/// the "right" eye is on the left of the image).
pub const RIGHT_EYE: usize = 0;
pub const LEFT_EYE: usize = 1;
pub const NOSE_TIP: usize = 2;
pub const RIGHT_MOUTH_CORNER: usize = 3;
pub const LEFT_MOUTH_CORNER: usize = 4;

/// One detected face. Coordinates are normalized [0,1] to the network input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceDetection {
    /// `[xmin, ymin, xmax, ymax]`, clamped to the image.
    pub bbox: [f32; 4],
    pub confidence: f32,
    /// See [`RIGHT_EYE`] .. [`LEFT_MOUTH_CORNER`].
    pub landmarks: [(f32, f32); 5],
}

/// Prior box `(cx, cy, w, h)`, normalized.
type Prior = [f32; 4];

pub struct YuNetParser {
    width: u32,
    height: u32,
    priors: Vec<Prior>,
    /// Minimum score `sqrt(cls * iou)` to keep a face (model default 0.6).
    pub confidence_threshold: f32,
    /// NMS overlap above which the weaker face is dropped (model default 0.3).
    pub iou_threshold: f32,
    /// Maximum number of faces returned.
    pub max_faces: usize,
}

impl YuNetParser {
    /// Parser for a YuNet model with this input size (e.g. 640x360).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            priors: priors(width, height),
            confidence_threshold: 0.6,
            iou_threshold: 0.3,
            max_faces: 50,
        }
    }

    pub fn input_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Decode the `loc`, `conf` and `iou` layers of a YuNet `NNData`.
    pub fn parse(&self, nn: &NNData) -> Result<Vec<FaceDetection>> {
        let loc = nn.tensor("loc")?;
        let conf = nn.tensor("conf")?;
        let iou = nn.tensor("iou")?;
        Ok(self.decode(&loc.data, &conf.data, &iou.data))
    }

    /// Decode raw outputs: `loc` is `[N, 14]`, `conf` `[N, 2]` (background,
    /// face probabilities), `iou` `[N, 1]`, for the N priors of this input
    /// size. Mismatched lengths decode the common prefix.
    pub fn decode(&self, loc: &[f32], conf: &[f32], iou: &[f32]) -> Vec<FaceDetection> {
        let n = self
            .priors
            .len()
            .min(loc.len() / LOC_STRIDE)
            .min(conf.len() / 2)
            .min(iou.len());
        let mut faces: Vec<FaceDetection> = (0..n)
            .filter_map(|i| {
                let score = (conf[2 * i + 1].clamp(0.0, 1.0) * iou[i].clamp(0.0, 1.0)).sqrt();
                (score >= self.confidence_threshold).then(|| decode_one(&self.priors[i], &loc[i * LOC_STRIDE..][..LOC_STRIDE], score))
            })
            .collect();
        faces.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        let mut kept: Vec<FaceDetection> = Vec::new();
        for face in faces {
            if kept.len() >= self.max_faces {
                break;
            }
            if kept.iter().all(|k| iou_of(&k.bbox, &face.bbox) <= self.iou_threshold) {
                kept.push(face);
            }
        }
        kept
    }

    /// Package `faces` (decoded from `source`) as `ImgDetections` with label
    /// 0 and the 5 landmarks as keypoints, ready to send to the device, e.g.
    /// to `SpatialLocationCalculator.inputDetections` for 3D positions.
    pub fn to_img_detections(source: &NNData, faces: &[FaceDetection]) -> Result<ImgDetections> {
        let mut dets = ImgDetections::for_nn_data(source)?;
        for face in faces {
            dets.push(0, face.confidence, face.bbox, &face.landmarks)?;
        }
        Ok(dets)
    }
}

fn decode_one(prior: &Prior, loc: &[f32], confidence: f32) -> FaceDetection {
    let [pcx, pcy, pw, ph] = *prior;
    let cx = pcx + loc[0] * VARIANCE[0] * pw;
    let cy = pcy + loc[1] * VARIANCE[0] * ph;
    let w = pw * (loc[2] * VARIANCE[1]).exp();
    let h = ph * (loc[3] * VARIANCE[1]).exp();
    let bbox = [
        (cx - w / 2.0).clamp(0.0, 1.0),
        (cy - h / 2.0).clamp(0.0, 1.0),
        (cx + w / 2.0).clamp(0.0, 1.0),
        (cy + h / 2.0).clamp(0.0, 1.0),
    ];
    let mut landmarks = [(0.0, 0.0); 5];
    for (j, lm) in landmarks.iter_mut().enumerate() {
        *lm = (
            pcx + loc[4 + 2 * j] * VARIANCE[0] * pw,
            pcy + loc[5 + 2 * j] * VARIANCE[0] * ph,
        );
    }
    FaceDetection { bbox, confidence, landmarks }
}

/// Prior boxes in the model's output order: per feature map, row-major
/// cells, then each min size.
fn priors(width: u32, height: u32) -> Vec<Prior> {
    // Feature map sizes as in OpenCV's FaceDetectorYN: input halved twice
    // (rounding up then down), then halved (rounding down) per level.
    let mut fw = (width + 1) / 2 / 2;
    let mut fh = (height + 1) / 2 / 2;
    let (w, h) = (width as f32, height as f32);
    let mut priors = Vec::new();
    for (level, min_sizes) in MIN_SIZES.iter().enumerate() {
        fw /= 2;
        fh /= 2;
        let step = STEPS[level];
        for y in 0..fh {
            for x in 0..fw {
                for &min_size in min_sizes.iter() {
                    priors.push([
                        (x as f32 + 0.5) * step / w,
                        (y as f32 + 0.5) * step / h,
                        min_size / w,
                        min_size / h,
                    ]);
                }
            }
        }
    }
    priors
}

fn iou_of(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_counts_match_zoo_models() {
        // Row counts of the `loc` output of the Luxonis zoo YuNet variants.
        assert_eq!(priors(320, 240).len(), 4385);
        assert_eq!(priors(640, 360).len(), 13150);
    }

    #[test]
    fn decodes_and_suppresses_overlaps() {
        let parser = YuNetParser::new(320, 240);
        let n = parser.priors.len();
        let (mut loc, mut conf, mut iou) = (vec![0.0; n * LOC_STRIDE], vec![0.0; n * 2], vec![0.0; n]);
        // Two priors of the same cell (overlapping boxes) and one far away, mid-image
        // so no clamping: prior 3060 is cell (20, 25) of the stride-8 map.
        for (i, score) in [(0usize, 0.9f32), (1, 0.8), (3060, 0.7)] {
            conf[2 * i + 1] = score;
            iou[i] = score;
        }
        let faces = parser.decode(&loc, &conf, &iou);
        assert_eq!(faces.len(), 2);
        assert!((faces[0].confidence - 0.9).abs() < 1e-6);
        // Zero offsets: box centered on its prior, landmarks at the center.
        let p = parser.priors[3060];
        let f = faces[1];
        assert!(((f.bbox[0] + f.bbox[2]) / 2.0 - p[0]).abs() < 1e-5);
        assert_eq!(f.landmarks[NOSE_TIP], (p[0], p[1]));
        // An offset moves the box by variance * prior size.
        loc[3060 * LOC_STRIDE] = 1.0;
        let moved = parser.decode(&loc, &conf, &iou)[1];
        assert!(((moved.bbox[0] + moved.bbox[2]) / 2.0 - (p[0] + 0.1 * p[2])).abs() < 1e-5);
    }
}
