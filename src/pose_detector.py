from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

BODY_PART_NAMES = (
    "head",
    "left_arm",
    "right_arm",
    "upper_torso",
    "lower_torso",
    "left_leg",
    "right_leg",
)


@dataclass
class PersonPose:
    frame_idx: int
    bbox: tuple[float, float, float, float]
    keypoints: dict[str, tuple[float, float, float]]


@dataclass
class PersonTrack:
    track_id: int
    poses: dict[int, PersonPose] = field(default_factory=dict)
    last_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    last_frame_idx: int = -1

    def add_pose(self, pose: PersonPose) -> None:
        self.poses[pose.frame_idx] = pose
        self.last_bbox = pose.bbox
        self.last_frame_idx = pose.frame_idx

    @property
    def first_frame_idx(self) -> int:
        return min(self.poses) if self.poses else -1

    @property
    def display_name(self) -> str:
        return f"Person {self.track_id}"


@dataclass
class PoseAnalysis:
    tracks: list[PersonTrack]
    num_frames: int
    frame_size: tuple[int, int]


@dataclass
class BodyPartSignal:
    person_id: int
    body_part: str
    signal_at_coords: np.ndarray
    valid_frames: np.ndarray

    @property
    def label(self) -> str:
        return f"Person {self.person_id} {self.body_part}"


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO Pose requires the optional 'ultralytics' package. "
            "Install it in this environment before using Pose Body-Part mode."
        ) from exc
    return YOLO


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _keypoint_similarity(
    prev_pose: PersonPose,
    pose: PersonPose,
    scale: float,
    min_confidence: float = 0.25,
) -> float:
    distances: list[float] = []
    for name in COCO_KEYPOINT_NAMES:
        prev_x, prev_y, prev_conf = prev_pose.keypoints[name]
        x, y, conf = pose.keypoints[name]
        if prev_conf < min_confidence or conf < min_confidence:
            continue
        distances.append(float(np.hypot(prev_x - x, prev_y - y)))

    if not distances:
        return 0.0

    # A score of 1 means the visible pose keypoints stayed in place; 0 means
    # they moved farther than the previous person bbox scale.
    return max(0.0, 1.0 - (float(np.mean(distances)) / max(1.0, scale)))


def _track_match_score(track: PersonTrack, pose: PersonPose, frame_gap: int) -> float:
    iou = _bbox_iou(track.last_bbox, pose.bbox)
    tx, ty = _bbox_center(track.last_bbox)
    px, py = _bbox_center(pose.bbox)
    diag = max(1.0, np.hypot(track.last_bbox[2] - track.last_bbox[0], track.last_bbox[3] - track.last_bbox[1]))
    center_score = max(0.0, 1.0 - (np.hypot(tx - px, ty - py) / (diag * 2.0)))
    prev_pose = track.poses.get(track.last_frame_idx)
    keypoint_score = _keypoint_similarity(prev_pose, pose, diag) if prev_pose else 0.0
    age_penalty = min(frame_gap, 10) * 0.03
    return (0.25 * iou) + (0.25 * center_score) + (0.50 * keypoint_score) - age_penalty


def detect_people_in_frame(model, frame: np.ndarray, frame_idx: int, confidence: float) -> list[PersonPose]:
    result = model(frame, verbose=False, conf=confidence)[0]
    if result.boxes is None or result.keypoints is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()
    poses: list[PersonPose] = []
    for bbox, kpts in zip(boxes, keypoints):
        named_kpts = {
            name: (float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2]))
            for i, name in enumerate(COCO_KEYPOINT_NAMES)
        }
        poses.append(PersonPose(frame_idx=frame_idx, bbox=tuple(map(float, bbox)), keypoints=named_kpts))
    return poses


def analyze_pose_video(
    frames: list[np.ndarray],
    model_path: str = "yolo11n-pose.pt",
    confidence: float = 0.35,
    track_match_threshold: float = 0.15,
    max_frame_gap: int = 12,
    min_track_frames: int = 3,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> PoseAnalysis:
    if not frames:
        return PoseAnalysis([], 0, (0, 0))

    YOLO = require_ultralytics()
    model = YOLO(model_path)
    tracks: list[PersonTrack] = []
    next_track_id = 1

    for frame_idx, frame in enumerate(frames):
        if progress_callback:
            progress_callback(frame_idx / max(1, len(frames)))
        poses = detect_people_in_frame(model, frame, frame_idx, confidence)
        unmatched_tracks = set(range(len(tracks)))

        for pose in poses:
            best_track_idx: Optional[int] = None
            best_score = track_match_threshold
            for track_idx in list(unmatched_tracks):
                track = tracks[track_idx]
                frame_gap = frame_idx - track.last_frame_idx
                if frame_gap > max_frame_gap:
                    continue
                score = _track_match_score(track, pose, frame_gap)
                if score > best_score:
                    best_score = score
                    best_track_idx = track_idx

            if best_track_idx is None:
                track = PersonTrack(track_id=next_track_id)
                next_track_id += 1
                track.add_pose(pose)
                tracks.append(track)
            else:
                tracks[best_track_idx].add_pose(pose)
                unmatched_tracks.remove(best_track_idx)

    if progress_callback:
        progress_callback(1.0)

    h, w = frames[0].shape[:2]
    tracks = [track for track in tracks if len(track.poses) >= min_track_frames]
    return PoseAnalysis(tracks=tracks, num_frames=len(frames), frame_size=(w, h))


def _visible_points(pose: PersonPose, names: tuple[str, ...], min_confidence: float) -> Optional[np.ndarray]:
    pts: list[tuple[int, int]] = []
    for name in names:
        x, y, conf = pose.keypoints[name]
        if conf < min_confidence:
            return None
        pts.append((int(round(x)), int(round(y))))
    return np.array(pts, dtype=np.int32)


def _limb_thickness(points: np.ndarray) -> int:
    if len(points) < 2:
        return 1
    distances = [np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))]
    return max(3, int(np.mean(distances) * 0.28))


def body_part_mask(
    pose: PersonPose,
    body_part: str,
    frame_size: tuple[int, int],
    min_keypoint_confidence: float = 0.25,
) -> Optional[np.ndarray]:
    w, h = frame_size
    mask_u8 = np.zeros((h, w), dtype=np.uint8)

    match body_part:
        case "head":
            pts = _visible_points(pose, ("nose", "left_eye", "right_eye"), min_keypoint_confidence)
            if pts is None:
                return None
            cv2.fillConvexPoly(mask_u8, pts, 1)
        case "left_arm":
            pts = _visible_points(pose, ("left_shoulder", "left_elbow", "left_wrist"), min_keypoint_confidence)
            if pts is None:
                return None
            cv2.polylines(mask_u8, [pts], False, 1, thickness=_limb_thickness(pts))
        case "right_arm":
            pts = _visible_points(pose, ("right_shoulder", "right_elbow", "right_wrist"), min_keypoint_confidence)
            if pts is None:
                return None
            cv2.polylines(mask_u8, [pts], False, 1, thickness=_limb_thickness(pts))
        case "upper_torso":
            pts = _visible_points(pose, ("left_shoulder", "right_shoulder"), min_keypoint_confidence)
            if pts is None:
                return None
            cv2.line(mask_u8, tuple(pts[0]), tuple(pts[1]), 1, thickness=_limb_thickness(pts))
        case "lower_torso":
            pts = _visible_points(pose, ("left_hip", "right_hip"), min_keypoint_confidence)
            if pts is None:
                return None
            cv2.line(mask_u8, tuple(pts[0]), tuple(pts[1]), 1, thickness=_limb_thickness(pts))
        case "left_leg":
            pts = _visible_points(pose, ("left_hip", "left_knee", "left_ankle"), min_keypoint_confidence)
            if pts is None:
                return None
            cv2.polylines(mask_u8, [pts], False, 1, thickness=_limb_thickness(pts))
        case "right_leg":
            pts = _visible_points(pose, ("right_hip", "right_knee", "right_ankle"), min_keypoint_confidence)
            if pts is None:
                return None
            cv2.polylines(mask_u8, [pts], False, 1, thickness=_limb_thickness(pts))
        case _:
            raise ValueError(f"Unknown body part: {body_part}")

    return mask_u8.astype(bool)


def _interpolate_missing(signal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if np.all(valid):
        return signal
    if not np.any(valid):
        return np.zeros_like(signal)
    idx = np.arange(len(signal))
    filled = signal.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], signal[valid])
    return filled


def extract_body_part_signal(
    motion_data_matrix: np.ndarray,
    track: PersonTrack,
    body_part: str,
    min_keypoint_confidence: float = 0.25,
) -> BodyPartSignal:
    num_frames, h, w = motion_data_matrix.shape
    raw_signal = np.zeros(num_frames, dtype=float)
    valid = np.zeros(num_frames, dtype=bool)

    for frame_idx, pose in track.poses.items():
        if frame_idx >= num_frames:
            continue
        mask = body_part_mask(pose, body_part, (w, h), min_keypoint_confidence)
        if mask is None or not np.any(mask):
            continue
        raw_signal[frame_idx] = float(np.mean(motion_data_matrix[frame_idx][mask]))
        valid[frame_idx] = True

    signal = _interpolate_missing(raw_signal, valid)
    return BodyPartSignal(track.track_id, body_part, signal, valid)


def _pose_mean_confidence(pose: PersonPose) -> float:
    return float(np.mean([conf for _x, _y, conf in pose.keypoints.values()]))


def merge_person_tracks(analysis: PoseAnalysis, track_ids: set[int]) -> Optional[int]:
    if len(track_ids) < 2:
        return None

    tracks_to_merge = [track for track in analysis.tracks if track.track_id in track_ids]
    if len(tracks_to_merge) < 2:
        return None

    target_id = min(track.track_id for track in tracks_to_merge)
    merged_poses: dict[int, PersonPose] = {}
    for track in tracks_to_merge:
        for frame_idx, pose in track.poses.items():
            existing = merged_poses.get(frame_idx)
            if existing is None or _pose_mean_confidence(pose) > _pose_mean_confidence(existing):
                merged_poses[frame_idx] = pose

    merged_track = PersonTrack(track_id=target_id)
    for frame_idx in sorted(merged_poses):
        merged_track.add_pose(merged_poses[frame_idx])

    remaining_tracks = [track for track in analysis.tracks if track.track_id not in track_ids]
    remaining_tracks.append(merged_track)
    analysis.tracks = sorted(remaining_tracks, key=lambda track: track.track_id)
    return target_id


def draw_pose_overlay(
    rgb_frame: np.ndarray,
    track: PersonTrack,
    frame_idx: int,
    selected_body_parts: set[str],
    color: tuple[int, int, int] = (255, 64, 64),
) -> None:
    pose = track.poses.get(frame_idx)
    if pose is None:
        return
    x1, y1, x2, y2 = map(int, pose.bbox)
    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
    for body_part in selected_body_parts:
        mask = body_part_mask(pose, body_part, (rgb_frame.shape[1], rgb_frame.shape[0]))
        if mask is None:
            continue
        rgb_frame[mask] = (0.65 * rgb_frame[mask] + 0.35 * np.array(color)).astype(np.uint8)
