import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ----------------------------------------------------------------------------------------
# Settings (all can be tuned; most are also exposed as command-line options)
# ----------------------------------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480          # every eye image is cropped/resized to this

# Pupil detection (values from the JEO tracker)
THRESHOLD_OFFSETS = (25, 15, 5)      # relaxed, medium, strict: added to darkest pixel value
PUPIL_SEARCH_BOX = 250               # px square kept around the darkest patch
MIN_PUPIL_AREA = 1000                # px^2, smaller blobs are ignored
MAX_PUPIL_ASPECT = 3.0               # reject long thin blobs (eyelashes, shadows)
VALID_CONFIDENCE = 0.35              # below this the frame is treated as "no pupil" (--min-confidence)

# Eye-sphere model (values from the JEO tracker)
RAY_CONFIDENCE = 0.85                # only confident pupils feed the eye-centre estimate
RADIUS_CONFIDENCE = 0.90
MAX_RAYS = 100
MAX_INTERSECTIONS = 1500
CENTER_HISTORY = 200
RAYS_PER_ESTIMATE = 4
MIN_RAY_ANGLE_DEG = 8.0
INTERSECTION_AGREEMENT_PX = 30
MODEL_READY_CENTERS = 60             # centre estimates needed before the model is "ready"
MAX_ROUNDNESS_FOR_RAY = 0.97         # nearly circular pupils give no usable ray direction

# Movement classification (tune for your cameras; 30 fps only resolves larger saccades)
FIXATION_MAX_DEG_S = 15.0
SACCADE_MIN_DEG_S = 40.0
BLINK_MAX_S = 0.50                   # pupil missing longer than this = NO PUPIL, not a blink
DIRECTION_DEADZONE_DEG = 8.0
SMOOTHING_SAMPLES = 3

MAX_JUMP_PX = 120                    # pupil jumping further than this in one frame = detection glitch
MAX_AREA_CHANGE = 2.0                # pupil area changing more than 2x in one frame = detection glitch
MIN_BLINK_S = 0.08                   # shorter dropouts are detection gaps, not blinks

TRACE_SECONDS = 6.0
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker_settings.json")
PLOT_RANGE_DEG = 40.0


# ----------------------------------------------------------------------------------------
# Pupil detection (JEO Research method)
# ----------------------------------------------------------------------------------------
def crop_to_aspect_ratio(image, width=FRAME_W, height=FRAME_H):
    h, w = image.shape[:2]
    desired = width / height
    if w / h > desired:
        new_w = int(desired * h)
        off = (w - new_w) // 2
        image = image[:, off:off + new_w]
    else:
        new_h = int(w / desired)
        off = (h - new_h) // 2
        image = image[off:off + new_h, :]
    return cv2.resize(image, (width, height))


def get_darkest_area(gray):
    """Centre of the darkest 20x20 block (sparse sampling, same grid as the JEO code, vectorised)."""
    ignore, step, block, inner = 20, 10, 20, 5
    h, w = gray.shape
    ys = np.arange(ignore, h - ignore, step)
    xs = np.arange(ignore, w - ignore, step)
    if len(ys) == 0 or len(xs) == 0:
        return None
    g = gray.astype(np.int32)
    total = np.zeros((len(ys), len(xs)), np.int64)
    for dy in range(0, block, inner):
        for dx in range(0, block, inner):
            total += g[np.ix_(np.minimum(ys + dy, h - 1), np.minimum(xs + dx, w - 1))]
    iy, ix = np.unravel_index(np.argmin(total), total.shape)
    return int(xs[ix] + block // 2), int(ys[iy] + block // 2)


def mask_outside_square(image, center, size):
    x, y = center
    half = size // 2
    mask = np.zeros_like(image)
    mask[max(0, y - half):min(image.shape[0], y + half), max(0, x - half):min(image.shape[1], x + half)] = 255
    return cv2.bitwise_and(image, mask)


def largest_reasonable_contour(contours):
    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_PUPIL_AREA:
            continue
        _, _, w, h = cv2.boundingRect(c)
        if min(w, h) == 0 or max(w, h) / min(w, h) > MAX_PUPIL_ASPECT:
            continue
        if area > best_area:
            best, best_area = c, area
    return best


def ellipse_fill_ratio(binary, contour):
    """Share of the fitted ellipse's area that is actually dark (pupil) pixels."""
    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary)
    cv2.ellipse(mask, ellipse, 255, -1)
    area = np.count_nonzero(mask)
    if area == 0:
        return 0.0
    return np.count_nonzero((binary == 255) & (mask == 255)) / area


def contour_pixels_under_ellipse(contour, shape):
    """(pixels of the contour near the ellipse, fraction of contour pixels on the ellipse)."""
    contour_mask = np.zeros(shape, np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, 1)
    ellipse = cv2.fitEllipse(contour)
    thick = np.zeros(shape, np.uint8)
    thin = np.zeros(shape, np.uint8)
    cv2.ellipse(thick, ellipse, 255, 10)
    cv2.ellipse(thin, ellipse, 255, 4)
    border = np.count_nonzero(contour_mask)
    abs_thick = np.count_nonzero(cv2.bitwise_and(contour_mask, thick))
    abs_thin = np.count_nonzero(cv2.bitwise_and(contour_mask, thin))
    return abs_thick, (abs_thin / border if border else 0.0)


def optimize_contour_by_angle(contour):
    """Keep only contour points that curve toward the centroid (drops eyelid/eyelash bumps)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 5:
        return contour
    spacing = max(1, n // 25)
    prev_pts = np.roll(pts, spacing, axis=0)
    next_pts = np.roll(pts, -spacing, axis=0)
    v1 = prev_pts - pts
    v2 = next_pts - pts
    to_centroid = pts.mean(axis=0) - pts
    keep = np.einsum("ij,ij->i", to_centroid, (v1 + v2) / 2) >= math.cos(math.radians(60))
    return pts[keep].astype(np.int32).reshape(-1, 1, 2)


def detect_pupil(gray, want_debug=False):
    """Return (ellipse or None, confidence 0..1, darkest point, debug images)."""
    darkest = get_darkest_area(gray)
    if darkest is None:
        return None, 0.0, None, None
    darkest_value = int(gray[darkest[1], darkest[0]])
    kernel = np.ones((5, 5), np.uint8)

    best_score, best_contour, best_ratio = 0.0, None, 0.0
    debug = []
    for offset in THRESHOLD_OFFSETS:
        _, binary = cv2.threshold(gray, darkest_value + offset, 255, cv2.THRESH_BINARY_INV)
        binary = mask_outside_square(binary, darkest, PUPIL_SEARCH_BOX)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        if want_debug:
            debug.append(dilated)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = largest_reasonable_contour(contours)
        if contour is None or len(contour) <= 5:
            continue
        filled = ellipse_fill_ratio(dilated, contour)
        abs_px, ratio = contour_pixels_under_ellipse(contour, dilated.shape)
        score = filled * abs_px * abs_px * ratio
        if score > best_score:
            best_score, best_contour, best_ratio = score, contour, ratio

    if best_contour is None:
        return None, 0.0, darkest, debug
    refined = optimize_contour_by_angle(best_contour)
    if len(refined) < 5:
        return None, 0.0, darkest, debug
    ellipse = cv2.fitEllipse(refined)
    (cx, cy), (w, h), _ = ellipse
    if not (0 <= cx < gray.shape[1] and 0 <= cy < gray.shape[0]) or w <= 0 or h <= 0:
        return None, 0.0, darkest, debug
    return ellipse, float(best_ratio), darkest, debug


# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------
def minor_axis_ray(ellipse):
    """Line through the pupil centre along the ellipse's minor axis: (cx, cy, angle 0..180)."""
    (cx, cy), (w, h), angle = ellipse
    direction = angle if w <= h else angle + 90.0
    return cx, cy, direction % 180.0


def angle_gap(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def intersect_rays(r1, r2):
    x1, y1, a1 = r1
    x2, y2, a2 = r2
    d1 = np.array([math.cos(math.radians(a1)), math.sin(math.radians(a1))])
    d2 = np.array([math.cos(math.radians(a2)), math.sin(math.radians(a2))])
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t1, _ = np.linalg.solve(A, np.array([x2 - x1, y2 - y1]))
    return x1 + t1 * d1[0], y1 + t1 * d1[1]


def distance_to_pupil_outer_edge(eye_center, ellipse):
    (px, py), (w, h), angle = ellipse
    dx, dy = px - eye_center[0], py - eye_center[1]
    dist = math.hypot(dx, dy)
    a, b = w / 2, h / 2
    if dist == 0 or a <= 0 or b <= 0:
        return None
    ux, uy = dx / dist, dy / dist
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    lx, ly = c * ux + s * uy, -s * ux + c * uy
    return dist + 1 / math.sqrt((lx / a) ** 2 + (ly / b) ** 2)


def compute_gaze_vector(px, py, cx, cy, width=FRAME_W, height=FRAME_H):
    """JEO sphere model: pupil (px,py) and eye centre (cx,cy) in pixels -> unit gaze vector."""
    fov_y = math.radians(45.0)
    far = 100.0
    cam = np.array([0.0, 0.0, 3.0])
    half_h = math.tan(fov_y / 2) * far
    half_w = half_h * width / height
    ndc_x = 2.0 * px / width - 1.0
    ndc_y = 1.0 - 2.0 * py / height
    far_point = np.array([ndc_x * half_w, ndc_y * half_h, cam[2] - far])
    direction = far_point - cam
    direction /= np.linalg.norm(direction)

    radius = 1.0 / 1.05
    sphere = np.array([(cx / width * 2.0 - 1.0) * 1.5, (1.0 - cy / height * 2.0) * 1.5, 0.0])
    L = cam - sphere
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - radius ** 2
    disc = b * b - 4 * c
    t = -np.dot(direction, L)
    if disc >= 0:
        roots = [r for r in ((-b - math.sqrt(disc)) / 2, (-b + math.sqrt(disc)) / 2) if r > 0]
        if roots:
            t = min(roots)
    local = cam + t * direction - sphere
    norm = np.linalg.norm(local)
    gaze = local / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    return sphere, gaze


def rotation_from_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


# ----------------------------------------------------------------------------------------
# Per-eye 3D eye model (eye-sphere centre from intersecting pupil rays)
# ----------------------------------------------------------------------------------------
class EyeModel:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rays = deque(maxlen=MAX_RAYS)
        self.intersections = deque(maxlen=MAX_INTERSECTIONS)
        self.centers = deque(maxlen=CENTER_HISTORY)
        self.center = (FRAME_W // 2, FRAME_H // 2)
        self.radius = 0.0
        self.radius_ellipse = None
        self.locked = False

    @property
    def ready(self):
        return self.locked or len(self.centers) >= MODEL_READY_CENTERS

    def update(self, ellipse, confidence):
        if self.locked or ellipse is None:
            return
        (_, _), (w, h), _ = ellipse
        roundness = min(w, h) / max(w, h)
        if confidence >= RAY_CONFIDENCE and roundness < MAX_ROUNDNESS_FOR_RAY:
            self.rays.append(minor_axis_ray(ellipse))

        estimate = self._estimate_center()
        if estimate is not None:
            self.centers.append(estimate)
            self.center = (int(np.mean([p[0] for p in self.centers])),
                           int(np.mean([p[1] for p in self.centers])))

        if self.radius_ellipse is not None:
            d = distance_to_pupil_outer_edge(self.center, self.radius_ellipse)
            if d is not None:
                self.radius = d
        if confidence >= RADIUS_CONFIDENCE and len(self.centers) >= MODEL_READY_CENTERS:
            d = distance_to_pupil_outer_edge(self.center, ellipse)
            if d is not None and (self.radius_ellipse is None or d > self.radius):
                self.radius, self.radius_ellipse = d, ellipse

    def _estimate_center(self):
        if len(self.rays) < 2:
            return None
        chosen = random.sample(list(self.rays), min(RAYS_PER_ESTIMATE, len(self.rays)))
        points = []
        for r1, r2 in zip(chosen, chosen[1:]):
            if angle_gap(r1[2], r2[2]) < MIN_RAY_ANGLE_DEG:
                continue
            p = intersect_rays(r1, r2)
            if p and 0 <= p[0] < FRAME_W and 0 <= p[1] < FRAME_H:
                points.append(p)
        if not points:
            return None
        agree = all(math.dist(points[i], points[j]) <= INTERSECTION_AGREEMENT_PX
                    for i in range(len(points)) for j in range(i + 1, len(points)))
        if agree:
            self.intersections.extend(points)
        if not self.intersections:
            return None
        return (float(np.mean([p[0] for p in self.intersections])),
                float(np.mean([p[1] for p in self.intersections])))


# ----------------------------------------------------------------------------------------
# Movement analysis (angles, velocity, events) for one eye
# ----------------------------------------------------------------------------------------
class MotionAnalyzer:
    def __init__(self):
        self.invert_horizontal = False
        self.reset()

    def reset(self):
        self.zero_rotation = None
        self.raw_dirs = deque(maxlen=10)
        self.window = deque(maxlen=SMOOTHING_SAMPLES)
        self.prev = None                  # (t, yaw, pitch) of previous smoothed sample
        self.invalid_since = None
        self.blinks = 0
        self.saccades = 0
        self._in_saccade = False
        self.trace = deque()              # (t, yaw, pitch) for plotting
        self.event = "NO PUPIL"
        self.velocity = 0.0

    @property
    def zeroed(self):
        return self.zero_rotation is not None

    def set_zero(self):
        if not self.raw_dirs:
            return False
        mean = np.mean(np.array(self.raw_dirs), axis=0)
        self.zero_rotation = rotation_from_a_to_b(mean, np.array([0.0, 0.0, 1.0]))
        self.window.clear()
        self.prev = None
        return True

    def angles(self, gaze):
        g = self.zero_rotation @ gaze if self.zero_rotation is not None else gaze
        # Camera faces the eye, so +x in the image is the wearer's LEFT: negate for "+ = right".
        yaw = -math.degrees(math.atan2(g[0], g[2]))
        pitch = math.degrees(math.atan2(g[1], g[2]))
        if self.invert_horizontal:
            yaw = -yaw
        return yaw, pitch

    def update(self, t, valid, gaze):
        if not valid:
            if self.invalid_since is None:
                self.invalid_since = t
            missing = t - self.invalid_since
            self.event = "BLINK" if missing <= BLINK_MAX_S else "NO PUPIL"
            self.velocity = 0.0
            self.window.clear()
            self._in_saccade = False
            return None, None

        if self.invalid_since is not None:
            if MIN_BLINK_S <= t - self.invalid_since <= BLINK_MAX_S:
                self.blinks += 1
            self.invalid_since = None
            self.prev = None

        self.raw_dirs.append(gaze)
        yaw, pitch = self.angles(gaze)
        self.window.append((yaw, pitch))
        s_yaw = float(np.median([w[0] for w in self.window]))
        s_pitch = float(np.median([w[1] for w in self.window]))

        if self.prev is not None and 0 < t - self.prev[0] < 0.2:
            dt = t - self.prev[0]
            self.velocity = math.hypot(s_yaw - self.prev[1], s_pitch - self.prev[2]) / dt
        else:
            self.velocity = 0.0
        self.prev = (t, s_yaw, s_pitch)

        if self.velocity >= SACCADE_MIN_DEG_S:
            self.event = "SACCADE"
            if not self._in_saccade:
                self.saccades += 1
            self._in_saccade = True
        else:
            self._in_saccade = False
            self.event = "FIXATION" if self.velocity < FIXATION_MAX_DEG_S else "PURSUIT"

        self.trace.append((t, s_yaw, s_pitch))
        while self.trace and t - self.trace[0][0] > TRACE_SECONDS:
            self.trace.popleft()
        return s_yaw, s_pitch


def direction_label(yaw, pitch):
    if yaw is None:
        return "-"
    vertical = "UP" if pitch > DIRECTION_DEADZONE_DEG else "DOWN" if pitch < -DIRECTION_DEADZONE_DEG else ""
    horizontal = "RIGHT" if yaw > DIRECTION_DEADZONE_DEG else "LEFT" if yaw < -DIRECTION_DEADZONE_DEG else ""
    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    return vertical or horizontal or "CENTER"


# ----------------------------------------------------------------------------------------
# Frame sources: live USB camera (threaded, newest frame wins) or a video file
# ----------------------------------------------------------------------------------------
def backends():
    if sys.platform == "win32":
        return [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    if sys.platform == "darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION), ("Auto", cv2.CAP_ANY)]
    return [("V4L2", cv2.CAP_V4L2), ("Auto", cv2.CAP_ANY)]


def open_camera(index):
    for name, backend in backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # MJPG keeps two cameras on one USB controller within bandwidth.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, name
        cap.release()
    return None, None


class CameraSource:
    def __init__(self, index, label):
        self.label = label
        self.cap, backend = open_camera(index)
        if self.cap is None:
            raise RuntimeError(f"Could not open camera {index} for the {label} eye. "
                               f"Run with --list-cameras to see which numbers work.")
        print(f"{label} eye: camera {index} via {backend}")
        self.lock = threading.Lock()
        self.frame, self.t, self.frame_id = None, 0.0, 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame, self.t, self.frame_id = frame, time.perf_counter(), self.frame_id + 1

    def latest(self):
        with self.lock:
            return self.frame, self.t, self.frame_id

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class VideoSource:
    def __init__(self, path, label):
        self.label = label
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_id = 0
        self.frame = None
        self.ended = False

    def latest(self):
        ok, frame = self.cap.read()
        if not ok:
            self.ended = True
            return self.frame, self.frame_id / self.fps, self.frame_id
        self.frame_id += 1
        self.frame = frame
        return frame, self.frame_id / self.fps, self.frame_id

    def close(self):
        self.cap.release()


def apply_flip(frame, flip):
    if flip == "h":
        return cv2.flip(frame, 1)
    if flip == "v":
        return cv2.flip(frame, 0)
    if flip == "vh":
        return cv2.flip(frame, -1)
    return frame


# ----------------------------------------------------------------------------------------
# One eye: detection + model + motion, plus its drawing
# ----------------------------------------------------------------------------------------
class EyeTracker:
    def __init__(self, name, source, flip, mirror=False):
        self.name = name
        self.source = source
        self.flip = flip
        self.mirror = mirror            # glasses whose cameras see the eye mirrored
        self.prev_ellipse = None
        self.model = EyeModel()
        self.motion = MotionAnalyzer()
        self.last_id = -1
        self.sample = None
        self.view = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        self.debug_images = None
        self.frames = 0
        self.valid_frames = 0

    def step(self, want_debug=False):
        frame, t, frame_id = self.source.latest()
        if frame is None or frame_id == self.last_id:
            return None
        self.last_id = frame_id
        frame = apply_flip(frame, self.flip)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        frame = crop_to_aspect_ratio(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        ellipse, confidence, darkest, debug = detect_pupil(gray, want_debug)
        self.debug_images = debug
        valid = ellipse is not None and confidence >= VALID_CONFIDENCE
        if valid and self.prev_ellipse is not None and self._is_glitch(ellipse):
            valid = False
        self.prev_ellipse = ellipse if valid else None
        self.frames += 1
        self.valid_frames += int(valid)

        gaze = None
        if valid:
            self.model.update(ellipse, confidence)
            _, gaze = compute_gaze_vector(ellipse[0][0], ellipse[0][1], *self.model.center)
        yaw, pitch = self.motion.update(t, valid, gaze)

        self.sample = {
            "t": t, "valid": valid, "confidence": confidence, "ellipse": ellipse,
            "gaze": gaze, "yaw": yaw, "pitch": pitch,
            "velocity": self.motion.velocity, "event": self.motion.event,
            "direction": direction_label(yaw, pitch) if self.motion.zeroed else "-",
        }
        self.view = self._draw(frame, ellipse, valid, darkest)
        return self.sample

    def _is_glitch(self, ellipse):
        (x0, y0), (w0, h0), _ = self.prev_ellipse
        (x1, y1), (w1, h1), _ = ellipse
        jump = math.hypot(x1 - x0, y1 - y0)
        a0, a1 = max(w0 * h0, 1.0), max(w1 * h1, 1.0)
        return jump > MAX_JUMP_PX or max(a0, a1) / min(a0, a1) > MAX_AREA_CHANGE

    def set_mirror(self, mirror):
        self.mirror = mirror
        self.model.reset()
        self.motion.reset()
        self.prev_ellipse = None

    def _draw(self, frame, ellipse, valid, darkest):
        img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        center = self.model.center
        if self.model.ready:
            if self.model.radius > 0:
                cv2.circle(img, center, int(self.model.radius), (255, 80, 80), 2)
            cv2.circle(img, center, 6, (255, 255, 0), -1)
        if ellipse is not None:
            color = (0, 255, 0) if valid else (0, 0, 255)
            cv2.ellipse(img, ellipse, color, 2)
            px, py = int(ellipse[0][0]), int(ellipse[0][1])
            cv2.circle(img, (px, py), 3, color, -1)
            if valid and self.model.ready:
                ex, ey = int(center[0] + 2 * (px - center[0])), int(center[1] + 2 * (py - center[1]))
                cv2.line(img, (px, py), (ex, ey), (200, 255, 0), 3)
        elif darkest is not None:
            cv2.drawMarker(img, darkest, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

        s = self.sample
        lines = [f"{self.name.upper()} EYE   conf {s['confidence'] * 100:4.0f}%"
                 f"{'   mirrored' if self.mirror else ''}"]
        if not self.model.ready:
            lines.append("Model settling: look around slowly")
        elif not self.model.locked:
            lines.append("Model ready: press F to lock")
        if s["yaw"] is not None:
            tag = "" if self.motion.zeroed else "  (press Z to zero)"
            lines.append(f"H {s['yaw']:+6.1f} deg  V {s['pitch']:+6.1f} deg{tag}")
        lines.append(f"{s['event']:<9} {s['velocity']:6.0f} deg/s   {s['direction']}")
        box_h = 14 + 28 * len(lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_W, box_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, text in enumerate(lines):
            cv2.putText(img, text, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return img


def put_text(img, text, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------------------
# Dashboard drawing
# ----------------------------------------------------------------------------------------
EYE_COLORS = {"left": (255, 200, 0), "right": (255, 0, 220)}   # BGR: cyan-ish, magenta


def draw_gaze_plot(w, h, eyes):
    img = np.full((h, w, 3), 24, np.uint8)
    cx, cy = w // 2, h // 2
    scale = (min(w, h) / 2 - 12) / PLOT_RANGE_DEG
    for deg in (10, 20, 30, 40):
        cv2.circle(img, (cx, cy), int(deg * scale), (55, 55, 55), 1)
    cv2.line(img, (0, cy), (w, cy), (70, 70, 70), 1)
    cv2.line(img, (cx, 0), (cx, h), (70, 70, 70), 1)
    put_text(img, "gaze (H,V)  rings = 10 deg", (8, 18), 0.45, (190, 190, 190))
    for eye in eyes:
        color = EYE_COLORS[eye.name]
        clamp = lambda yaw, pitch: (int(np.clip(cx + yaw * scale, 4, w - 5)), int(np.clip(cy - pitch * scale, 4, h - 5)))
        pts = [clamp(y, p) for _, y, p in list(eye.motion.trace)[-20:]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 1, cv2.LINE_AA)
        s = eye.sample
        if s and s["yaw"] is not None:
            cv2.circle(img, clamp(s["yaw"], s["pitch"]), 7, color, -1)
    return img


def draw_trace(w, h, eyes, now):
    img = np.full((h, w, 3), 18, np.uint8)
    mid = h // 2
    scale = (h / 2 - 10) / PLOT_RANGE_DEG
    for deg in (-20, 0, 20):
        y = int(mid - deg * scale)
        cv2.line(img, (0, y), (w, y), (60, 60, 60) if deg else (90, 90, 90), 1)
        put_text(img, f"{deg:+d}", (w - 34, y - 3), 0.4, (150, 150, 150))
    put_text(img, f"horizontal angle, last {TRACE_SECONDS:.0f} s (+ = right)", (8, 18), 0.45, (190, 190, 190))
    for eye in eyes:
        pts, last_t = [], None
        for t, yaw, _ in eye.motion.trace:
            x = int(w - (now - t) / TRACE_SECONDS * w)
            pt = (x, int(np.clip(mid - yaw * scale, 2, h - 3)))
            if last_t is not None and t - last_t < 0.2:
                cv2.line(img, pts[-1], pt, EYE_COLORS[eye.name], 2, cv2.LINE_AA)
            pts.append(pt)
            last_t = t
    return img


def compose_dashboard(eyes, status_lines, now, panel_w=480):
    panel_h = panel_w * FRAME_H // FRAME_W
    panels = [cv2.resize(e.view, (panel_w, panel_h)) for e in eyes]
    top = np.hstack(panels) if len(panels) > 1 else panels[0]
    total_w = top.shape[1]
    plot_h = 220
    plot = draw_gaze_plot(plot_h, plot_h, eyes)
    trace = draw_trace(total_w - plot_h, plot_h, eyes, now)
    bottom = np.hstack([plot, trace])
    status = np.full((22 * len(status_lines) + 12, total_w, 3), 32, np.uint8)
    for i, text in enumerate(status_lines):
        put_text(status, text, (10, 22 + 22 * i), 0.5)
    return np.vstack([top, bottom, status])


# ----------------------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------------------
CSV_FIELDS = ["time_s", "eye", "valid", "confidence", "pupil_x", "pupil_y", "pupil_w", "pupil_h",
              "pupil_angle", "eye_center_x", "eye_center_y", "gaze_x", "gaze_y", "gaze_z",
              "h_deg", "v_deg", "velocity_deg_s", "event", "direction", "model_locked", "zeroed"]


class Recorder:
    def __init__(self, folder):
        self.folder = folder
        self.file = None
        self.writer = None
        self.t0 = None
        self.path = None

    @property
    def active(self):
        return self.file is not None

    def start(self):
        os.makedirs(self.folder, exist_ok=True)
        self.path = os.path.join(self.folder, datetime.now().strftime("eye_log_%Y%m%d_%H%M%S.csv"))
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(CSV_FIELDS)
        self.t0 = None
        print(f"Recording to {self.path}")

    def stop(self):
        if self.file:
            self.file.close()
            print(f"Saved {self.path}")
        self.file = self.writer = None

    def write(self, eye):
        s = eye.sample
        if not self.active or s is None:
            return
        if self.t0 is None:
            self.t0 = s["t"]
        e = s["ellipse"]
        g = s["gaze"]
        r = lambda v, n=3: "" if v is None else round(float(v), n)
        self.writer.writerow([
            r(s["t"] - self.t0, 4), eye.name, int(s["valid"]), r(s["confidence"]),
            r(e[0][0], 2) if e else "", r(e[0][1], 2) if e else "",
            r(e[1][0], 2) if e else "", r(e[1][1], 2) if e else "", r(e[2], 1) if e else "",
            eye.model.center[0], eye.model.center[1],
            r(g[0], 5) if g is not None else "", r(g[1], 5) if g is not None else "",
            r(g[2], 5) if g is not None else "",
            r(s["yaw"], 2), r(s["pitch"], 2), r(s["velocity"], 1), s["event"], s["direction"],
            int(eye.model.locked), int(eye.motion.zeroed),
        ])


# ----------------------------------------------------------------------------------------
# Camera listing
# ----------------------------------------------------------------------------------------
def list_cameras(max_index=8):
    found = []
    for i in range(max_index):
        cap, backend = open_camera(i)
        if cap is None:
            continue
        ok, frame = cap.read()
        cap.release()
        if ok:
            found.append((i, frame))
            print(f"camera {i}: {frame.shape[1]}x{frame.shape[0]} via {backend}")
    if not found:
        print("No cameras found. Check the USB cables, then try again.")
        return
    tiles = []
    for i, frame in found:
        tile = cv2.resize(frame, (320, 240))
        cv2.rectangle(tile, (0, 0), (150, 42), (0, 0, 0), -1)
        cv2.putText(tile, f"camera {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
    cv2.imshow("Cameras - note the two eye-camera numbers, press any key", np.vstack(rows))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\nThen run:  python dual_eye_tracker.py --left <number> --right <number>")


# ----------------------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------------------
def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(**values):
    settings = load_settings()
    settings.update(values)
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except OSError as exc:
        print(f"Could not save settings: {exc}")


def parse_args():
    p = argparse.ArgumentParser(description="Binocular eye-movement detector (JEO Research method).")
    p.add_argument("--list-cameras", action="store_true", help="show all cameras with their numbers")
    p.add_argument("--left", type=int, help="camera number of the LEFT-eye camera")
    p.add_argument("--right", type=int, help="camera number of the RIGHT-eye camera")
    p.add_argument("--left-video", help="use a recorded video instead of the left camera")
    p.add_argument("--right-video", help="use a recorded video instead of the right camera")
    p.add_argument("--flip-left", choices=["none", "h", "v", "vh"], default=None,
                   help="flip the left image: h = mirror, v = upside down, vh = both")
    p.add_argument("--flip-right", choices=["none", "h", "v", "vh"], default=None)
    p.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None,
                   help="un-mirror both eye images (for cameras that see the eye mirrored); "
                        "--no-mirror turns it off. Also toggled with the M key.")
    p.add_argument("--record", action="store_true", help="start CSV recording immediately")
    p.add_argument("--out", default="recordings", help="folder for CSV and video output")
    p.add_argument("--save-video", action="store_true", help="also save the dashboard as .mp4")
    p.add_argument("--saccade", type=float, default=SACCADE_MIN_DEG_S, help="saccade threshold, deg/s")
    p.add_argument("--fixation", type=float, default=FIXATION_MAX_DEG_S, help="fixation threshold, deg/s")
    p.add_argument("--min-confidence", type=float, default=VALID_CONFIDENCE,
                   help="0-1; raise it if eyelashes are mistaken for the pupil, lower it if the pupil is often missed")
    p.add_argument("--no-display", action="store_true", help="run without windows (for testing)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    return p.parse_args()


def main():
    global SACCADE_MIN_DEG_S, FIXATION_MAX_DEG_S, VALID_CONFIDENCE
    args = parse_args()
    SACCADE_MIN_DEG_S, FIXATION_MAX_DEG_S = args.saccade, args.fixation
    VALID_CONFIDENCE = args.min_confidence

    if args.list_cameras:
        list_cameras()
        return

    # Camera numbers, flips and mirroring are remembered between runs, so after the first
    # run "python dual_eye_tracker.py" on its own is enough.
    saved = load_settings()
    using_cameras = not (args.left_video or args.right_video)
    if using_cameras and args.left is None and args.right is None and saved:
        args.left, args.right = saved.get("left"), saved.get("right")
        if args.left is not None or args.right is not None:
            print(f"Using saved cameras: left {args.left}, right {args.right} "
                  f"(change with --left / --right)")
    flip_left = args.flip_left or saved.get("flip_left", "none")
    flip_right = args.flip_right or saved.get("flip_right", "none")
    mirror = args.mirror if args.mirror is not None else saved.get("mirror", False)
    if using_cameras and (args.left is not None or args.right is not None):
        save_settings(left=args.left, right=args.right, flip_left=flip_left,
                      flip_right=flip_right, mirror=mirror)

    specs = [("left", args.left, args.left_video, flip_left),
             ("right", args.right, args.right_video, flip_right)]
    eyes = []
    try:
        for name, index, video, flip in specs:
            if video:
                source = VideoSource(video, name)
            elif index is not None:
                source = CameraSource(index, name)
            else:
                continue
            eyes.append(EyeTracker(name, source, flip, mirror))
    except RuntimeError as exc:
        print(exc)
        for e in eyes:
            e.source.close()
        return
    if not eyes:
        print("Tell me which cameras to use, e.g.  python dual_eye_tracker.py --left 1 --right 2\n"
              "Not sure of the numbers? Run:      python dual_eye_tracker.py --list-cameras")
        return

    recorder = Recorder(args.out)
    if args.record:
        recorder.start()
    video_writer = None
    debug = paused = False
    window = "Dual Eye Tracker"
    processed = 0
    start = time.perf_counter()
    fps_times = deque(maxlen=30)

    try:
        while True:
            if not paused:
                updated = False
                for eye in eyes:
                    if eye.step(want_debug=debug) is not None:
                        recorder.write(eye)
                        updated = True
                if updated:
                    processed += 1
                    fps_times.append(time.perf_counter())
                if all(isinstance(e.source, VideoSource) and e.source.ended for e in eyes):
                    break

            now = max((e.sample["t"] for e in eyes if e.sample), default=0.0)
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) if len(fps_times) > 1 else 0.0
            status = [f"{fps:4.1f} fps   rec: {'ON  ' + os.path.basename(recorder.path) if recorder.active else 'off'}"
                      f"   image: {'mirrored (M to undo)' if eyes[0].mirror else 'as camera sees it (M to mirror)'}"
                      f"{'   PAUSED' if paused else ''}"]
            stats = []
            for e in eyes:
                rate = 100 * e.valid_frames / e.frames if e.frames else 0
                stats.append(f"{e.name}: detect {rate:3.0f}%  blinks {e.motion.blinks}  "
                             f"saccades {e.motion.saccades}  model {'LOCKED' if e.model.locked else 'auto'}")
            status.append("   |   ".join(stats))
            if len(eyes) == 2 and all(e.sample and e.sample["yaw"] is not None for e in eyes):
                l, r = eyes[0].sample, eyes[1].sample
                status.append(f"both eyes: H {(l['yaw'] + r['yaw']) / 2:+.1f}  V {(l['pitch'] + r['pitch']) / 2:+.1f}"
                              f"   L-R difference: H {l['yaw'] - r['yaw']:+.1f}  V {l['pitch'] - r['pitch']:+.1f} deg")
            else:
                status.append("both eyes: -")
            status.append("keys: F lock  Z zero (look straight)  S record  R reset  M mirror  "
                          "D debug  1/2 swap L-R  SPACE pause  Q quit")
            dashboard = compose_dashboard(eyes, status, now)

            if args.save_video:
                if video_writer is None:
                    os.makedirs(args.out, exist_ok=True)
                    vpath = os.path.join(args.out, datetime.now().strftime("eye_video_%Y%m%d_%H%M%S.mp4"))
                    video_writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                                   (dashboard.shape[1], dashboard.shape[0]))
                    print(f"Saving video to {vpath}")
                video_writer.write(dashboard)

            if args.max_frames and processed >= args.max_frames:
                break
            if args.no_display:
                continue

            cv2.imshow(window, dashboard)
            if debug:
                for e in eyes:
                    if e.debug_images:
                        tiles = [cv2.resize(d, (320, 240)) for d in e.debug_images]
                        cv2.imshow(f"{e.name} thresholds (relaxed | medium | strict)", np.hstack(tiles))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("f"), ord("F")):
                for e in eyes:
                    e.model.locked = not e.model.locked
                print("Eye model", "LOCKED" if eyes[0].model.locked else "automatic")
            elif key in (ord("z"), ord("Z")):
                done = [e.name for e in eyes if e.motion.set_zero()]
                print("Zeroed:", ", ".join(done) if done else "no eye visible - try again")
            elif key in (ord("s"), ord("S")):
                recorder.stop() if recorder.active else recorder.start()
            elif key in (ord("r"), ord("R")):
                for e in eyes:
                    e.model.reset()
                    e.motion.reset()
                print("Eye models reset - look around to rebuild them")
            elif key in (ord("m"), ord("M")):
                mirror = not eyes[0].mirror
                for e in eyes:
                    e.set_mirror(mirror)
                if using_cameras:
                    save_settings(mirror=mirror)
                print("Images", "mirrored" if mirror else "not mirrored",
                      "- eye models reset, look around again, then F and Z")
            elif key in (ord("d"), ord("D")):
                debug = not debug
                if not debug:
                    cv2.destroyAllWindows()
            elif key in (ord("1"), ord("2")):
                idx = 0 if key == ord("1") else 1
                if idx < len(eyes):
                    m = eyes[idx].motion
                    m.invert_horizontal = not m.invert_horizontal
                    m.window.clear()
                    m.prev = None
                    print(f"{eyes[idx].name} eye: left/right swapped")
    finally:
        recorder.stop()
        if video_writer is not None:
            video_writer.release()
        for e in eyes:
            e.source.close()
        cv2.destroyAllWindows()
        elapsed = time.perf_counter() - start
        for e in eyes:
            rate = 100 * e.valid_frames / e.frames if e.frames else 0
            print(f"{e.name}: {e.frames} frames, pupil found in {rate:.0f}%, "
                  f"{e.motion.blinks} blinks, {e.motion.saccades} saccades")
        print(f"ran {elapsed:.1f} s")


if __name__ == "__main__":
    main()
