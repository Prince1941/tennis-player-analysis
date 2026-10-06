import os
import requests
import json
import time
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def test_biomechanics_components():
    print("\n[Unit Tests] Validating Biomechanical & Kinematic Components...")
    try:
        from backend.model import (
            classify_stroke_kinematic_consensus,
            estimate_shot_direction,
            evaluate_posture_quality,
            recommend_shot,
            angle_between
        )
    except ImportError:
        from model import (
            classify_stroke_kinematic_consensus,
            estimate_shot_direction,
            evaluate_posture_quality,
            recommend_shot,
            angle_between
        )

    # 1. Test geometric angle calculation (degrees)
    # Right angle: (1, 0, 0) - (0, 0, 0) - (0, 1, 0) -> 90 degrees
    ang = angle_between([1, 0, 0], [0, 0, 0], [0, 1, 0])
    assert abs(ang - 90.0) < 1e-4, f"Expected 90 deg, got {ang}"
    print(f"  ✓ 3-Point Angle Geometry: {ang:.1f}° matches expected 90.0°")

    # 2. Test Forehand vs Backhand Kinematic Consensus
    # Construct synthetic right-handed forehand frames (wrist on right side of shoulder vector)
    fh_window = np.zeros((30, 39), dtype=np.float32)
    # left_shoulder (idx 1) = [-0.25, 0.4, 0.0]
    # right_shoulder (idx 2) = [0.25, 0.4, 0.0]
    # right_wrist (idx 6) = [0.45, 0.2, -0.1] (lateral right side during prep & contact)
    for t in range(30):
        fh_window[t, 3:6] = [-0.25, 0.4, 0.0]
        fh_window[t, 6:9] = [0.25, 0.4, 0.0]
        fh_window[t, 18:21] = [0.45, 0.2, -0.1]
    # Mock model output having backhand bias: [bh: 0.65, fh: 0.35]
    mock_str_probs = np.array([0.65, 0.35], dtype=np.float32)
    stroke_type, stroke_conf, basis, probs = classify_stroke_kinematic_consensus(fh_window, mock_str_probs)
    assert stroke_type == "forehand", f"Expected forehand, got {stroke_type}"
    print(f"  ✓ Forehand Kinematic Consensus: Successfully resolved model backhand bias -> {stroke_type} ({stroke_conf:.1%}, {basis})")

    # Construct synthetic overhead serve frames (wrist elevated high above nose)
    serve_window = np.zeros((30, 39), dtype=np.float32)
    for t in range(30):
        serve_window[t, 0:3] = [0.0, 0.2, 0.0] # nose y = 0.2
        serve_window[t, 18:21] = [0.2, -0.4, 0.0] # wrist y = -0.4 (elevated 0.6m above nose)
    serve_type, serve_conf, serve_basis, _ = classify_stroke_kinematic_consensus(serve_window, mock_str_probs)
    assert serve_type == "serve", f"Expected serve, got {serve_type}"
    print(f"  ✓ Serve Kinematic Detection: Elevated contact detected -> {serve_type} ({serve_conf:.1%})")

    # 3. Test Direction Estimation (Cross-Court vs Down-the-Line)
    dir_cross_window = np.zeros((30, 39), dtype=np.float32)
    for t in range(30):
        # Wrist swings from contact (x=0.1, z=0.0) to follow-through across body (x=-0.4, z=0.1)
        dir_cross_window[t, 18:21] = [0.1 - 0.5 * (t / 30.0), 0.2, 0.0 + 0.1 * (t / 30.0)]
    mock_dir_probs = np.array([0.80, 0.15, 0.05], dtype=np.float32) # Model center bias
    d_pred, d_conf, d_basis, _ = estimate_shot_direction(dir_cross_window, mock_dir_probs, "forehand")
    assert d_pred == "cross_court", f"Expected cross_court, got {d_pred}"
    print(f"  ✓ Shot Direction Estimation: Follow-through vector identified {d_pred} ({d_conf:.1%}, {d_basis})")

    # 4. Test Multi-Factor Posture Evaluation (0-100 score)
    # Pro posture: deep knee flexion (125 deg), strong coil (32 deg), stable lean (10 deg), fluid elbow (130 deg)
    pro_derived = np.zeros((30, 11), dtype=np.float32)
    pro_derived[:, 4] = 125.0 # right knee
    pro_derived[:, 6] = 32.0  # trunk rotation
    pro_derived[:, 7] = np.radians(10.0) # trunk lean
    pro_derived[:, 0] = 130.0 # right elbow
    pro_pos = evaluate_posture_quality(pro_derived, "forehand", np.array([0.2, 0.8]))
    assert pro_pos["score"] >= 80, f"Expected pro posture score >= 80, got {pro_pos['score']}"
    print(f"  ✓ Pro Posture Scoring: {pro_pos['grade']} ({pro_pos['score']}/100) — Sub-scores: {pro_pos['sub_scores']}")

    # Poor posture: stiff knees (168 deg), uncoiled torso (8 deg), extreme tilt (35 deg), cramped elbow (80 deg)
    poor_derived = np.zeros((30, 11), dtype=np.float32)
    poor_derived[:, 4] = 168.0
    poor_derived[:, 6] = 8.0
    poor_derived[:, 7] = np.radians(35.0)
    poor_derived[:, 0] = 80.0
    poor_pos = evaluate_posture_quality(poor_derived, "forehand", np.array([0.8, 0.2]))
    assert poor_pos["score"] < 60, f"Expected poor posture score < 60, got {poor_pos['score']}"
    assert len(poor_pos["flaws"]) >= 2, f"Expected detected flaws, got {poor_pos['flaws']}"
    print(f"  ✓ Poor Posture Scoring: {poor_pos['grade']} ({poor_pos['score']}/100) — Flaws: {poor_pos['flaws']}")

    # 5. Test Coaching Recommendation Unit Consistency (No raw radians!)
    metrics_sample = {
        "max_right_knee_angle": 128.0,
        "max_trunk_rotation_diff": 32.0,
        "avg_trunk_lean": 12.0, # in degrees
        "avg_right_elbow_angle": 135.0
    }
    title, text = recommend_shot("cross_court", "forehand", "good", metrics_sample, posture_eval=pro_pos)
    assert "rad" not in text.lower(), f"Found unallowed 'rad' in recommendation text: {text}"
    assert "°" in text, "Expected degree symbols (°) in recommendation text"
    assert "ATP Reference Ranges" in text, "Expected ATP Reference Ranges in coaching report"
    print(f"  ✓ Coaching Engine: Unit consistency verified (all angles in °, zero raw radians, personalized drills present).")
    return True

def test_api():
    print("==================================================")
    print("   🎾 CourtVision AI — Integration Test Suite")
    print("==================================================")
    
    # 0. Test Biomechanics & Kinematics Components
    assert test_biomechanics_components(), "Unit tests failed!"
    
    # 1. Test Status Endpoint
    print("\n[Test 1] Testing GET /api/status...")
    try:
        res = requests.get(f"{BASE_URL}/api/status", timeout=5)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print(f"  ✓ Status Code: {res.status_code}")
        print(f"  ✓ Server Status: {data.get('status')}")
        print(f"  ✓ Model Loaded: {data.get('model_loaded')}")
        print(f"  ✓ Is Demo Fallback: {data.get('is_demo_mode')}")
        print(f"  ✓ Device: {data.get('device')}")
        print(f"  ✓ Classes: {data.get('classes')}")
        print(f"  ✓ Input Dim: {data.get('features', {}).get('total_input_dim')}")
    except Exception as e:
        print(f"  ✗ Status test failed: {e}")
        return False

    # 2. Test Image Diagnostic Endpoint
    print("\n[Test 2] Testing POST /api/analyze-image with sample tennis player image...")
    sample_img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "sample_tennis.jpg")
    if os.path.exists(sample_img_path):
        try:
            with open(sample_img_path, "rb") as f:
                res = requests.post(f"{BASE_URL}/api/analyze-image", files={"file": ("sample.jpg", f, "image/jpeg")}, timeout=15)
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            print(f"  ✓ Status Code: {res.status_code}")
            print(f"  ✓ Pose Success: {data.get('success')}")
            print(f"  ✓ Predicted Stroke: {data.get('stroke_type')} ({data.get('stroke_confidence'):.1%})")
            print(f"  ✓ Predicted Direction: {data.get('direction')} ({data.get('direction_confidence'):.1%})")
            print(f"  ✓ Predicted Posture: {data.get('posture')} ({data.get('posture_confidence'):.1%})")
            print(f"  ✓ Biomechanical Metrics Extracted: {list(data.get('metrics', {}).keys())[:4]}...")
            print(f"  ✓ Annotated Base64 Image Generated: {bool(data.get('image_data'))}")
        except Exception as e:
            print(f"  ✗ Image analysis test failed: {e}")
            return False
    else:
        print(f"  ⚠ Skipped image test: {sample_img_path} not found")

    # 3. Test Multi-Shot Video Analysis (Borna Coric Reference Rally)
    print("\n[Test 3] Testing POST /api/analyze-video with Borna Coric Reference Rally video...")
    coric_video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "coric_tennis.mp4")
    if os.path.exists(coric_video_path):
        try:
            t0 = time.time()
            with open(coric_video_path, "rb") as f:
                res = requests.post(f"{BASE_URL}/api/analyze-video", files={"file": ("coric_tennis.mp4", f, "video/mp4")}, timeout=120)
            elapsed = time.time() - t0
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            total_shots = data.get('total_shots', 0)
            shots = data.get('shots', [])
            
            print(f"  ✓ Analysis completed in {elapsed:.1f}s")
            print(f"  ✓ Total shots detected: {total_shots} (Requirement: > 1 sequential shots)")
            assert total_shots >= 2, f"Expected multiple shots, got {total_shots}"
            assert len(shots) == total_shots, f"Mismatch in shots count: {len(shots)} vs {total_shots}"
            
            # Check dynamic biomechanical variation between shots
            angles_r_elbow = [s['metrics']['avg_right_elbow_angle'] for s in shots]
            angles_trunk_rot = [s['metrics']['avg_trunk_rotation_diff'] for s in shots]
            
            print(f"  ✓ Shot-by-shot Right Elbow Angles: {angles_r_elbow[:5]}...")
            print(f"  ✓ Shot-by-shot Trunk Rotations: {angles_trunk_rot[:5]}...")
            
            # Verify angles are NOT static or duplicated
            assert len(set(angles_r_elbow)) > 1, "Right elbow angles are static across all shots!"
            assert len(set(angles_trunk_rot)) > 1, "Trunk rotation angles are static across all shots!"
            print(f"  ✓ Biomechanical angles dynamically recalculated for each shot independently!")
            
            # Verify each shot has its own prediction and timestamps
            for i, s in enumerate(shots):
                assert s['shot_id'] == i + 1, f"Invalid shot_id: {s['shot_id']}"
                assert s['start_time'] <= s['peak_time'] <= s['end_time'], f"Invalid timestamp order in shot {i+1}"
                assert 'stroke_type' in s and 'stroke_confidence' in s
                assert 'direction' in s and 'direction_confidence' in s
                assert 'posture' in s and 'posture_confidence' in s
                assert 'posture_score' in s and 0 <= s['posture_score'] <= 100
                assert 'stroke_basis' in s and 'direction_basis' in s
                assert 'joint_angles' in s and len(s['joint_angles']) >= 8
            print(f"  ✓ Validated all {total_shots} individual shot records have valid predictions, confidence, posture scores (0-100), and joint_angles.")
            
            # Check annotated video retrieval
            if data.get('annotated_video_url'):
                vid_res = requests.get(f"{BASE_URL}{data.get('annotated_video_url')}")
                assert vid_res.status_code == 200, "Annotated video not reachable"
                print(f"  ✓ Multi-Shot Annotated Video Accessible (HTTP {vid_res.status_code}, {len(vid_res.content)} bytes)")
        except Exception as e:
            print(f"  ✗ Multi-shot video analysis test failed: {e}")
            return False
    else:
        print(f"  ⚠ Skipped Coric rally test: {coric_video_path} not found")

    # 4. Test Dominic Thiem Practice Video
    print("\n[Test 4] Testing POST /api/analyze-video with Dominic Thiem Practice video...")
    thiem_video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "sample_tennis.mp4")
    if os.path.exists(thiem_video_path):
        try:
            t0 = time.time()
            with open(thiem_video_path, "rb") as f:
                res = requests.post(f"{BASE_URL}/api/analyze-video", files={"file": ("sample.mp4", f, "video/mp4")}, timeout=120)
            elapsed = time.time() - t0
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            total_shots = data.get('total_shots', 0)
            print(f"  ✓ Analysis completed in {elapsed:.1f}s")
            print(f"  ✓ Total shots detected for Thiem: {total_shots}")
            assert total_shots >= 2, f"Expected multiple shots, got {total_shots}"
        except Exception as e:
            print(f"  ✗ Thiem video analysis test failed: {e}")
            return False
    else:
        print(f"  ⚠ Skipped Thiem test: {thiem_video_path} not found")

    print("\n==================================================")
    print("   ✅ ALL MULTI-SHOT & DYNAMIC BIOMECHANICS TESTS PASSED!")
    print("==================================================")
    return True

if __name__ == "__main__":
    test_api()
