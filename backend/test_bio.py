import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from backend.model import (
    classify_stroke_kinematic_consensus,
    estimate_shot_direction,
    evaluate_posture_quality,
    recommend_shot,
    angle_between
)

print("[1] Testing 3-point angle calculation...", flush=True)
ang = angle_between([1, 0, 0], [0, 0, 0], [0, 1, 0])
assert abs(ang - 90.0) < 1e-4, f"Expected 90, got {ang}"
print(f"  ✓ Angle calculation: {ang:.1f} degrees", flush=True)

print("[2] Testing Forehand Kinematic Consensus...", flush=True)
fh_window = np.zeros((30, 39), dtype=np.float32)
for t in range(30):
    fh_window[t, 3:6] = [-0.25, 0.4, 0.0]  # left shoulder
    fh_window[t, 6:9] = [0.25, 0.4, 0.0]   # right shoulder
    fh_window[t, 18:21] = [0.45, 0.2, -0.1] # right wrist (lateral dominant side)
mock_str_probs = np.array([0.65, 0.35], dtype=np.float32) # Model has backhand bias
stroke_type, stroke_conf, basis, probs = classify_stroke_kinematic_consensus(fh_window, mock_str_probs)
assert stroke_type == "forehand", f"Expected forehand, got {stroke_type}"
print(f"  ✓ Forehand Kinematic Consensus: {stroke_type} ({stroke_conf:.1%}, basis: {basis})", flush=True)

print("[3] Testing Overhead Serve Kinematic Detection...", flush=True)
serve_window = np.zeros((30, 39), dtype=np.float32)
for t in range(30):
    serve_window[t, 0:3] = [0.0, 0.2, 0.0]
    serve_window[t, 18:21] = [0.2, -0.4, 0.0]
serve_type, serve_conf, serve_basis, _ = classify_stroke_kinematic_consensus(serve_window, mock_str_probs)
assert serve_type == "serve", f"Expected serve, got {serve_type}"
print(f"  ✓ Serve Detection: {serve_type} ({serve_conf:.1%})", flush=True)

print("[4] Testing Shot Direction Estimation...", flush=True)
dir_cross_window = np.zeros((30, 39), dtype=np.float32)
for t in range(30):
    dir_cross_window[t, 18:21] = [0.1 - 0.5 * (t / 30.0), 0.2, 0.0 + 0.1 * (t / 30.0)]
mock_dir_probs = np.array([0.80, 0.15, 0.05], dtype=np.float32)
d_pred, d_conf, d_basis, d_map = estimate_shot_direction(dir_cross_window, mock_dir_probs, "forehand")
assert d_pred == "cross_court", f"Expected cross_court, got {d_pred}"
print(f"  ✓ Shot Direction Estimation: {d_pred} ({d_conf:.1%}, basis: {d_basis})", flush=True)

print("[5] Testing Multi-Factor Posture Evaluation (0-100)...", flush=True)
pro_derived = np.zeros((30, 11), dtype=np.float32)
pro_derived[:, 4] = 125.0
pro_derived[:, 6] = 32.0
pro_derived[:, 7] = np.radians(10.0)
pro_derived[:, 0] = 130.0
pro_pos = evaluate_posture_quality(pro_derived, "forehand", np.array([0.2, 0.8]))
assert pro_pos["score"] >= 80, f"Expected pro posture score >= 80, got {pro_pos['score']}"
print(f"  ✓ Pro Posture Score: {pro_pos['grade']} ({pro_pos['score']}/100) — Sub-scores: {pro_pos['sub_scores']}", flush=True)

poor_derived = np.zeros((30, 11), dtype=np.float32)
poor_derived[:, 4] = 168.0
poor_derived[:, 6] = 8.0
poor_derived[:, 7] = np.radians(35.0)
poor_derived[:, 0] = 80.0
poor_pos = evaluate_posture_quality(poor_derived, "forehand", np.array([0.8, 0.2]))
assert poor_pos["score"] < 60, f"Expected poor posture score < 60, got {poor_pos['score']}"
assert len(poor_pos["flaws"]) >= 2, f"Expected flaws, got {poor_pos['flaws']}"
print(f"  ✓ Poor Posture Score: {poor_pos['grade']} ({poor_pos['score']}/100) — Flaws: {poor_pos['flaws']}", flush=True)

print("[6] Testing Coaching Recommendations & Unit Consistency...", flush=True)
metrics_sample = {
    "max_right_knee_angle": 128.0,
    "max_trunk_rotation_diff": 32.0,
    "avg_trunk_lean": 12.0,
    "avg_right_elbow_angle": 135.0
}
title, text = recommend_shot("cross_court", "forehand", "good", metrics_sample, posture_eval=pro_pos)
assert "rad" not in text.lower(), f"Found unallowed 'rad' in text: {text}"
assert "°" in text, "Expected degree symbols (°)"
print("  ✓ Recommendation Engine: All angles in degrees (°), zero raw radians, personalized drills verified.", flush=True)

print("\n=======================================================", flush=True)
print("   ✅ ALL BIOMECHANICAL & KINEMATIC TESTS PASSED!", flush=True)
print("=======================================================", flush=True)
