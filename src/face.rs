//! Face recognition helpers: landmark-based face alignment for embedding
//! models such as ArcFace (Luxonis model zoo `arcface:lfw-112x112`), and
//! embedding comparison.
//!
//! Recognition models expect faces warped so the eyes, nose and mouth land
//! on fixed positions. [`align_face`] crops that warp from a frame using the
//! 5 landmarks of a face detector (e.g. [`crate::parsers::yunet`]); send the
//! result to the embedding network with [`crate::queue::InputQueue::send_frame`].

/// Landmark positions of the standard 112x112 ArcFace crop, in the order
/// right eye, left eye, nose tip, right and left mouth corners (as seen by
/// the subject, like [`crate::parsers::yunet::FaceDetection::landmarks`]).
pub const ARCFACE_112_TEMPLATE: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

/// Least-squares similarity transform (rotation, uniform scale, translation)
/// mapping `from` points onto `to` points, as a 2x3 matrix
/// `[a, -b, tx, b, a, ty]`: `x' = a*x - b*y + tx`, `y' = b*x + a*y + ty`.
pub fn similarity_transform(from: &[(f32, f32)], to: &[(f32, f32)]) -> [f32; 6] {
    let n = from.len().min(to.len()).max(1) as f32;
    let mean = |pts: &[(f32, f32)]| {
        let (sx, sy) = pts.iter().fold((0.0, 0.0), |(sx, sy), &(x, y)| (sx + x, sy + y));
        (sx / n, sy / n)
    };
    let (fx, fy) = mean(from);
    let (tx, ty) = mean(to);
    let (mut dot, mut cross, mut norm) = (0.0, 0.0, 0.0);
    for (&(x, y), &(u, v)) in from.iter().zip(to) {
        let (x, y, u, v) = (x - fx, y - fy, u - tx, v - ty);
        dot += x * u + y * v;
        cross += x * v - y * u;
        norm += x * x + y * y;
    }
    let (a, b) = if norm > 0.0 { (dot / norm, cross / norm) } else { (1.0, 0.0) };
    [a, -b, tx - (a * fx - b * fy), b, a, ty - (b * fx + a * fy)]
}

/// Warp the face with `landmarks` (pixel coordinates in `frame`) onto the
/// 112x112 ArcFace template.
///
/// `frame` is planar (e.g. `BGR888p`, as a YuNet passthrough frame), with
/// `channels` planes of `width * height` bytes; the output has the same
/// layout at 112x112, with black outside the source image.
pub fn align_face(frame: &[u8], width: usize, height: usize, channels: usize, landmarks: &[(f32, f32); 5]) -> Vec<u8> {
    const SIZE: usize = 112;
    // Map output pixels back into the source frame.
    let m = similarity_transform(&ARCFACE_112_TEMPLATE, landmarks);
    let plane = width * height;
    let mut out = vec![0u8; SIZE * SIZE * channels];
    if frame.len() < plane * channels {
        return out;
    }
    for v in 0..SIZE {
        for u in 0..SIZE {
            let (uf, vf) = (u as f32, v as f32);
            let x = m[0] * uf + m[1] * vf + m[2];
            let y = m[3] * uf + m[4] * vf + m[5];
            if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
                continue;
            }
            // Bilinear interpolation.
            let (x0, y0) = (x as usize, y as usize);
            let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
            let (dx, dy) = (x - x0 as f32, y - y0 as f32);
            for c in 0..channels {
                let p = &frame[c * plane..][..plane];
                let top = p[y0 * width + x0] as f32 * (1.0 - dx) + p[y0 * width + x1] as f32 * dx;
                let bottom = p[y1 * width + x0] as f32 * (1.0 - dx) + p[y1 * width + x1] as f32 * dx;
                out[c * SIZE * SIZE + v * SIZE + u] = (top * (1.0 - dy) + bottom * dy).round() as u8;
            }
        }
    }
    out
}

/// Scale `v` to unit length (no-op for a zero vector).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Cosine similarity of two embeddings, in [-1, 1]; for ArcFace, the same
/// person typically scores above ~0.4 and different people below ~0.3.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_transform_recovers_known_transform() {
        // Rotate 30 degrees, scale 2, translate (5, -3).
        let (s, c) = (30f32.to_radians().sin() * 2.0, 30f32.to_radians().cos() * 2.0);
        let to: Vec<(f32, f32)> = ARCFACE_112_TEMPLATE.iter().map(|&(x, y)| (c * x - s * y + 5.0, s * x + c * y - 3.0)).collect();
        let m = similarity_transform(&ARCFACE_112_TEMPLATE, &to);
        for (got, want) in m.iter().zip([c, -s, 5.0, s, c, -3.0]) {
            assert!((got - want).abs() < 1e-3, "{m:?}");
        }
    }

    #[test]
    fn align_face_samples_landmarks_onto_template() {
        // A frame whose single channel encodes x coordinate; landmarks placed
        // at twice the template scale, so the crop samples x = 2 * u.
        let (w, h) = (256, 256);
        let frame: Vec<u8> = (0..w * h).map(|i| (i % w) as u8).collect();
        let landmarks = ARCFACE_112_TEMPLATE.map(|(x, y)| (2.0 * x, 2.0 * y));
        let out = align_face(&frame, w, h, 1, &landmarks);
        assert_eq!(out.len(), 112 * 112);
        assert_eq!(out[50 * 112 + 40], 80);
    }

    #[test]
    fn cosine_similarity_basics() {
        assert!((cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        let mut v = [3.0, 4.0];
        l2_normalize(&mut v);
        assert_eq!(v, [0.6, 0.8]);
    }
}
