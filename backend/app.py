import os
import sys
from unittest.mock import MagicMock
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import shutil
import tempfile
import logging
import base64
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
_mp_pose = None

def get_mp_pose():
    global _mp_pose
    if _mp_pose is None:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose
    return _mp_pose

# Robust directory paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
DEMO_DIR = os.path.join(ROOT_DIR, "demo")
SAMPLE_VIDEOS_DIR = os.path.join(ROOT_DIR, "sample videos")
MODEL_PATH = os.path.join(ROOT_DIR, "model_random_split.pt")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# Imports from backend.model
try:
    from backend.model import (
        get_model,
        compute_derived_features,
        normalize_and_resample_sequence,
        classify_stroke_kinematic_consensus,
        estimate_shot_direction,
        evaluate_posture_quality,
        recommend_shot,
        LANDMARK_INDEX_MAP,
        RELEVANT_LANDMARKS,
        CLASSES,
        SEQUENCE_LEN
    )
except ImportError:
    from model import (
        get_model,
        compute_derived_features,
        normalize_and_resample_sequence,
        classify_stroke_kinematic_consensus,
        estimate_shot_direction,
        evaluate_posture_quality,
        recommend_shot,
        LANDMARK_INDEX_MAP,
        RELEVANT_LANDMARKS,
        CLASSES,
        SEQUENCE_LEN
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tennis Biomechanics AI API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model initialization at startup (singleton)
model, is_random_fallback, model_device = get_model(MODEL_PATH, force_cpu=True)

# Static files and frontend routes
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>CourtVision AI Backend Running</h1>")

@app.get("/style.css")
def get_css():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))

@app.get("/app.js")
def get_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.js"))

@app.get("/api/videos/{filename}")
def get_video(filename: str):
    file_path = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Annotated video not found.")

@app.get("/demo/{filename}")
def get_demo_asset(filename: str):
    file_path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(file_path) and os.path.exists(SAMPLE_VIDEOS_DIR):
        # Fallback to sample videos folder
        alt_path = os.path.join(SAMPLE_VIDEOS_DIR, filename)
        if os.path.exists(alt_path):
            file_path = alt_path

    if os.path.exists(file_path):
        media_type = "video/mp4" if filename.endswith(".mp4") else "image/jpeg"
        return FileResponse(file_path, media_type=media_type)
    raise HTTPException(status_code=404, detail=f"Demo asset '{filename}' not found.")

# Response Models
class ShotAnalysis(BaseModel):
    shot_id: int
    peak_frame: int
    start_frame: int
    end_frame: int
    start_time: float
    peak_time: float
    end_time: float
    direction: str
    direction_confidence: float
    direction_basis: Optional[str] = "Kinematic Follow-Through Vector (Estimated)"
    direction_probabilities: Optional[Dict[str, float]] = None
    stroke_type: str
    stroke_confidence: float
    stroke_basis: Optional[str] = "Kinematic Trajectory Consensus"
    stroke_probabilities: Optional[Dict[str, float]] = None
    posture: str
    posture_confidence: float
    posture_score: Optional[int] = 75
    posture_grade: Optional[str] = "Good"
    flaws: Optional[List[str]] = None
    sub_scores: Optional[Dict[str, int]] = None
    posture_probabilities: Optional[Dict[str, float]] = None
    recommendation_title: str
    recommendation_text: str
    metrics: Dict[str, float]
    joint_angles: Optional[Dict[str, float]] = None
    posture_quality: Optional[Dict[str, Any]] = None

class BiomechanicsResponse(BaseModel):
    success: bool
    message: str
    is_demo: bool
    total_shots: int
    shots: List[ShotAnalysis]
    # Primary/Initial shot fields for backwards compatibility
    direction: str
    direction_confidence: float
    direction_probabilities: Optional[Dict[str, float]] = None
    stroke_type: str
    stroke_confidence: float
    stroke_probabilities: Optional[Dict[str, float]] = None
    posture: str
    posture_confidence: float
    posture_probabilities: Optional[Dict[str, float]] = None
    recommendation_title: str
    recommendation_text: str
    metrics: Dict[str, float]
    pose_history: List[Any]
    frame_metrics: Optional[List[Dict[str, float]]] = None
    annotated_video_url: Optional[str] = None
    stroke_peak_frame: Optional[int] = None
    total_frames_analyzed: Optional[int] = None

@app.get("/api/health")
@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "is_demo_mode": is_random_fallback,
        "model_path": MODEL_PATH,
        "device": str(model_device),
        "classes": CLASSES,
        "features": {
            "base_features": 39,
            "derived_features": 11,
            "total_input_dim": 50,
            "sequence_len": SEQUENCE_LEN
        }
    }

MAX_FILE_SIZE = 150 * 1024 * 1024  # 150 MB

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds maximum limit of 150MB.")
        
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
        
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_pose = get_mp_pose()
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(rgb_img)
        
    if not results.pose_landmarks:
        return {
            "success": False,
            "message": "No human pose detected in the image.",
            "metrics": {}
        }
        
    lm = results.pose_landmarks.landmark
    lm_world = results.pose_world_landmarks.landmark if results.pose_world_landmarks else lm
    
    # Extract 39 base features (world metric coords)
    base_features = []
    for landmark_name in RELEVANT_LANDMARKS:
        idx = LANDMARK_INDEX_MAP[landmark_name]
        pt = lm_world[idx]
        base_features.extend([pt.x, pt.y, pt.z])
        
    derived = compute_derived_features(base_features)
    
    metrics = {
        "right_elbow_angle": round(derived[0], 1),
        "left_elbow_angle": round(derived[1], 1),
        "right_shoulder_angle": round(derived[2], 1),
        "left_shoulder_angle": round(derived[3], 1),
        "right_knee_angle": round(derived[4], 1),
        "left_knee_angle": round(derived[5], 1),
        "trunk_rotation_diff": round(derived[6], 1),
        "trunk_lean": round(float(abs(np.degrees(derived[7]))), 1), # In degrees (no raw radians)
        "stance_width": round(derived[8], 3),
    }

    # Run model on repeated sequence with micro-noise (Cell 51 of research notebook)
    simulated_seq = [
        np.array(base_features) + np.random.normal(0, 0.001, len(base_features))
        for _ in range(SEQUENCE_LEN)
    ]
    resampled_features, derived_history = normalize_and_resample_sequence(simulated_seq, SEQUENCE_LEN)
    
    model.eval()
    input_tensor = torch.tensor(resampled_features, dtype=torch.float32).unsqueeze(0).to(model_device)
    with torch.no_grad():
        dir_logits, str_logits, pos_logits = model(input_tensor)
        dir_probs = F.softmax(dir_logits, dim=1).squeeze(0).cpu().numpy()
        str_probs = F.softmax(str_logits, dim=1).squeeze(0).cpu().numpy()
        pos_probs = F.softmax(pos_logits, dim=1).squeeze(0).cpu().numpy()

    # Kinematic consensus classification
    stroke_pred, str_conf, stroke_basis, str_prob_map = classify_stroke_kinematic_consensus(simulated_seq, str_probs)
    direction_pred, dir_conf, direction_basis, dir_prob_map = estimate_shot_direction(simulated_seq, dir_probs, stroke_pred)
    posture_eval = evaluate_posture_quality(derived_history, stroke_pred, pos_probs)
    posture_pred = posture_eval["grade"].lower()
    posture_conf = posture_eval["confidence"]

    rec_title, rec_text = recommend_shot(direction_pred, stroke_pred, posture_pred, metrics, posture_eval=posture_eval)

    # Draw skeletal overlay on image
    annotated_img = img.copy()
    connections = [
        (11,12), (11,13), (13,15), (12,14), (14,16),
        (11,23), (12,24), (23,24), (23,25), (24,26),
        (25,27), (26,28)
    ]
    for a, b in connections:
        pt1 = (int(lm[a].x * w), int(lm[a].y * h))
        pt2 = (int(lm[b].x * w), int(lm[b].y * h))
        cv2.line(annotated_img, pt1, pt2, (0, 255, 127), 2)
        
    for name, idx in LANDMARK_INDEX_MAP.items():
        pt = (int(lm[idx].x * w), int(lm[idx].y * h))
        cv2.circle(annotated_img, pt, 5, (0, 100, 255), -1)
        
    _, buffer = cv2.imencode('.jpg', annotated_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "success": True,
        "message": "Pose detected and analyzed successfully.",
        "metrics": metrics,
        "direction": direction_pred,
        "direction_confidence": dir_conf,
        "direction_basis": direction_basis,
        "direction_probabilities": dir_prob_map,
        "stroke_type": stroke_pred,
        "stroke_confidence": str_conf,
        "stroke_basis": stroke_basis,
        "stroke_probabilities": str_prob_map,
        "posture": posture_pred,
        "posture_confidence": posture_conf,
        "posture_score": posture_eval["score"],
        "posture_grade": posture_eval["grade"],
        "posture_quality": posture_eval,
        "recommendation_title": rec_title,
        "recommendation_text": rec_text,
        "image_data": f"data:image/jpeg;base64,{encoded_image}"
    }

@app.post("/api/analyze-video", response_model=BiomechanicsResponse)
async def analyze_video(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        file_size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File size exceeds maximum limit of 150MB.")
                buffer.write(chunk)
            
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot read uploaded video.")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        pose_history = []
        base_features_history = []
        frame_metrics_history = []
        
        last_valid_pose = None
        last_valid_base = None
        
        logger.info(f"Extracting pose landmarks and joint angles across all frames ({width}x{height} @ {fps:.1f} FPS)...")
        mp_pose = get_mp_pose()
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    lm_world = results.pose_world_landmarks.landmark if results.pose_world_landmarks else lm
                    
                    frame_pose = {}
                    for name, idx in LANDMARK_INDEX_MAP.items():
                        frame_pose[name] = {"x": lm[idx].x, "y": lm[idx].y, "z": lm[idx].z}
                    
                    # Base features from world metric landmarks (Cell 35 of research notebook)
                    base_f = []
                    for name in RELEVANT_LANDMARKS:
                        idx = LANDMARK_INDEX_MAP[name]
                        pt = lm_world[idx]
                        base_f.extend([pt.x, pt.y, pt.z])
                        
                    last_valid_pose = frame_pose
                    last_valid_base = base_f
                
                # Maintain temporal continuity
                if last_valid_pose is not None:
                    pose_history.append(last_valid_pose)
                    base_features_history.append(last_valid_base)
                    # Dynamic frame-level biomechanical calculation
                    d_feat = compute_derived_features(last_valid_base)
                else:
                    pose_history.append({name: {"x": 0.5, "y": 0.5, "z": 0.0} for name in RELEVANT_LANDMARKS})
                    base_features_history.append([0.0] * 39)
                    d_feat = [0.0] * 11
                    
                frame_metrics_history.append({
                    "right_elbow_angle": round(d_feat[0], 1),
                    "left_elbow_angle": round(d_feat[1], 1),
                    "right_shoulder_angle": round(d_feat[2], 1),
                    "left_shoulder_angle": round(d_feat[3], 1),
                    "right_knee_angle": round(d_feat[4], 1),
                    "left_knee_angle": round(d_feat[5], 1),
                    "trunk_rotation_diff": round(d_feat[6], 1),
                    "trunk_lean": round(d_feat[7], 3),
                    "stance_width": round(d_feat[8], 3)
                })
                    
        cap.release()
        total_frames = len(frames)
        
        if total_frames < 10 or len(base_features_history) < 10:
            raise HTTPException(status_code=400, detail="Video is too short or pose not detected.")
            
        base_arr = np.array(base_features_history, dtype=np.float32)
        
        # Kinetic velocity curve calculation (Cell 19 & Cell 62 of notebook)
        lw_idx = RELEVANT_LANDMARKS.index('left_wrist') * 3
        rw_idx = RELEVANT_LANDMARKS.index('right_wrist') * 3
        le_idx = RELEVANT_LANDMARKS.index('left_elbow') * 3
        re_idx = RELEVANT_LANDMARKS.index('right_elbow') * 3
        ls_idx = RELEVANT_LANDMARKS.index('left_shoulder') * 3
        rs_idx = RELEVANT_LANDMARKS.index('right_shoulder') * 3
        
        lw_vy = np.abs(np.diff(base_arr[:, lw_idx+1], prepend=base_arr[0, lw_idx+1]))
        rw_vy = np.abs(np.diff(base_arr[:, rw_idx+1], prepend=base_arr[0, rw_idx+1]))
        le_vy = np.abs(np.diff(base_arr[:, le_idx+1], prepend=base_arr[0, le_idx+1]))
        re_vy = np.abs(np.diff(base_arr[:, re_idx+1], prepend=base_arr[0, re_idx+1]))
        ls_vy = np.abs(np.diff(base_arr[:, ls_idx+1], prepend=base_arr[0, ls_idx+1]))
        rs_vy = np.abs(np.diff(base_arr[:, rs_idx+1], prepend=base_arr[0, rs_idx+1]))
        
        wrist_vel = np.maximum(lw_vy, rw_vy)
        elbow_vel = np.maximum(le_vy, re_vy)
        shoulder_vel = np.maximum(ls_vy, rs_vy)
        
        stroke_score = (wrist_vel * 0.5) + (elbow_vel * 0.3) + (shoulder_vel * 0.2)
        # 5-frame rolling mean to smooth high-frequency jitter
        stroke_score_smooth = np.convolve(stroke_score, np.ones(5)/5.0, mode='same')
        
        # Adaptive peak detection threshold & minimum stroke gap
        # Matches Cell 16 & 62 logic: identify every distinct stroke event
        min_gap_frames = max(20, int(fps * 0.85))  # at least ~0.85-1.0s cooldown between impact peaks
        mean_score = float(np.mean(stroke_score_smooth))
        std_score = float(np.std(stroke_score_smooth))
        velocity_threshold = max(0.018, mean_score + 0.3 * std_score)
        
        detected_peaks = []
        last_stroke_frame = -min_gap_frames
        
        for i in range(2, total_frames - 2):
            s = stroke_score_smooth[i]
            prev_s1 = stroke_score_smooth[i-1]
            prev_s2 = stroke_score_smooth[i-2]
            next_s1 = stroke_score_smooth[i+1]
            next_s2 = stroke_score_smooth[i+2]
            
            is_local_max = (s > prev_s1) and (s > prev_s2) and (s > next_s1) and (s > next_s2)
            above_thresh = s >= velocity_threshold
            cooldown_ok = (i - last_stroke_frame) >= min_gap_frames
            
            if is_local_max and above_thresh and cooldown_ok:
                detected_peaks.append(i)
                last_stroke_frame = i

        # Fallback if no peaks exceeded threshold: select global maximum
        if not detected_peaks:
            global_peak = int(np.argmax(stroke_score_smooth))
            detected_peaks.append(global_peak)

        logger.info(f"Detected {len(detected_peaks)} stroke event(s) at frames: {detected_peaks}")

        # Process EVERY shot independently
        shots: List[ShotAnalysis] = []
        WINDOW = 15

        for shot_idx, peak_frame in enumerate(detected_peaks):
            start_frame = max(0, peak_frame - WINDOW)
            end_frame = min(total_frames, peak_frame + WINDOW)
            
            start_time = round(float(start_frame / fps), 2)
            peak_time = round(float(peak_frame / fps), 2)
            end_time = round(float(end_frame / fps), 2)
            
            # Slicing temporal window around this stroke peak
            stroke_base_window = base_arr[start_frame:end_frame]
            norm_sequence, derived_history = normalize_and_resample_sequence(stroke_base_window, SEQUENCE_LEN)
            
            # Independent model inference for this shot
            model.eval()
            input_tensor = torch.tensor(norm_sequence, dtype=torch.float32).unsqueeze(0).to(model_device)
            
            with torch.no_grad():
                dir_logits, str_logits, pos_logits = model(input_tensor)
                dir_probs = F.softmax(dir_logits, dim=1).squeeze(0).cpu().numpy()
                str_probs = F.softmax(str_logits, dim=1).squeeze(0).cpu().numpy()
                pos_probs = F.softmax(pos_logits, dim=1).squeeze(0).cpu().numpy()
                
            # 1. Kinematic consensus stroke classification (resolving training dataset bias)
            shot_str, shot_str_conf, stroke_basis, str_prob_map = classify_stroke_kinematic_consensus(stroke_base_window, str_probs)

            # 2. Kinematic follow-through vector shot direction estimation
            shot_dir, shot_dir_conf, direction_basis, dir_prob_map = estimate_shot_direction(stroke_base_window, dir_probs, shot_str)

            # 3. Multi-factor 0-100 posture evaluation
            posture_eval = evaluate_posture_quality(derived_history, shot_str, pos_probs)
            shot_pos = posture_eval["grade"].lower()
            shot_pos_conf = posture_eval["confidence"]
            pos_prob_map = {CLASSES['posture'][i]: round(float(pos_probs[i]), 3) for i in range(len(CLASSES['posture']))}

            # Independent biomechanical joint-angle aggregation for THIS shot (strictly in degrees)
            shot_biomechanics = {
                "avg_right_elbow_angle": float(np.mean(derived_history[:, 0])),
                "max_right_elbow_angle": float(np.max(derived_history[:, 0])),
                "min_right_elbow_angle": float(np.min(derived_history[:, 0])),
                "avg_left_elbow_angle": float(np.mean(derived_history[:, 1])),
                "max_left_elbow_angle": float(np.max(derived_history[:, 1])),
                "avg_right_shoulder_angle": float(np.mean(derived_history[:, 2])),
                "avg_left_shoulder_angle": float(np.mean(derived_history[:, 3])),
                "avg_right_knee_angle": float(np.mean(derived_history[:, 4])),
                "max_right_knee_angle": float(np.max(derived_history[:, 4])),
                "min_right_knee_angle": float(np.min(derived_history[:, 4])),
                "avg_left_knee_angle": float(np.mean(derived_history[:, 5])),
                "avg_trunk_rotation_diff": float(np.mean(derived_history[:, 6])),
                "max_trunk_rotation_diff": float(np.max(derived_history[:, 6])),
                "avg_trunk_lean": float(round(abs(np.degrees(np.mean(derived_history[:, 7]))), 1)), # Strictly in degrees!
                "avg_stance_width": float(np.mean(derived_history[:, 8]))
            }
            formatted_shot_metrics = {k: round(v, 1) for k, v in shot_biomechanics.items()}
            
            rec_title, rec_text = recommend_shot(shot_dir, shot_str, shot_pos, formatted_shot_metrics, posture_eval=posture_eval)
            
            shot_joint_angles = {
                "right_elbow": formatted_shot_metrics.get("avg_right_elbow_angle", 0.0),
                "left_elbow": formatted_shot_metrics.get("avg_left_elbow_angle", 0.0),
                "right_shoulder": formatted_shot_metrics.get("avg_right_shoulder_angle", 0.0),
                "left_shoulder": formatted_shot_metrics.get("avg_left_shoulder_angle", 0.0),
                "right_knee": formatted_shot_metrics.get("avg_right_knee_angle", 0.0),
                "left_knee": formatted_shot_metrics.get("avg_left_knee_angle", 0.0),
                "trunk_rotation": formatted_shot_metrics.get("avg_trunk_rotation_diff", 0.0),
                "trunk_lean": formatted_shot_metrics.get("avg_trunk_lean", 0.0)
            }

            shots.append(ShotAnalysis(
                shot_id=shot_idx + 1,
                peak_frame=int(peak_frame),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                start_time=start_time,
                peak_time=peak_time,
                end_time=end_time,
                direction=shot_dir,
                direction_confidence=shot_dir_conf,
                direction_basis=direction_basis,
                direction_probabilities=dir_prob_map,
                stroke_type=shot_str,
                stroke_confidence=shot_str_conf,
                stroke_basis=stroke_basis,
                stroke_probabilities=str_prob_map,
                posture=shot_pos,
                posture_confidence=shot_pos_conf,
                posture_score=posture_eval["score"],
                posture_grade=posture_eval["grade"],
                flaws=posture_eval["flaws"],
                sub_scores=posture_eval["sub_scores"],
                posture_probabilities=pos_prob_map,
                recommendation_title=rec_title,
                recommendation_text=rec_text,
                metrics=formatted_shot_metrics,
                joint_angles=shot_joint_angles,
                posture_quality={
                    "prediction": posture_eval["grade"],
                    "score": posture_eval["score"],
                    "confidence": shot_pos_conf,
                    "flaws": posture_eval["flaws"],
                    "sub_scores": posture_eval["sub_scores"]
                }
            ))

        # Default/primary shot (the peak with highest velocity or first shot)
        primary_shot = max(shots, key=lambda s: stroke_score_smooth[s.peak_frame])

        # Generate Annotated Video with sequential dynamic HUD overlay
        output_filename = f"annotated_{int(time.time())}.mp4"
        annotated_video_path = os.path.join(OUTPUTS_DIR, output_filename)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(annotated_video_path, fourcc, min(fps, 30.0), (width, height))
            
            connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
            ]
            
            for i, frm in enumerate(frames):
                ann = frm.copy()
                landmarks = pose_history[i]
                
                # Draw skeleton bones
                for ptA, ptB in connections:
                    if ptA in landmarks and ptB in landmarks:
                        p1 = landmarks[ptA]
                        p2 = landmarks[ptB]
                        x1, y1 = int(p1['x'] * width), int(p1['y'] * height)
                        x2, y2 = int(p2['x'] * width), int(p2['y'] * height)
                        if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                            cv2.line(ann, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                # Draw joint nodes
                for joint_name, p in landmarks.items():
                    jx, jy = int(p['x'] * width), int(p['y'] * height)
                    if 0 <= jx < width and 0 <= jy < height:
                        cv2.circle(ann, (jx, jy), 5, (235, 99, 37), -1)

                # Determine active shot for this frame
                active_shot = None
                for s in shots:
                    if s.start_frame <= i <= s.end_frame:
                        active_shot = s
                        break
                if active_shot is None:
                    # Show nearest shot
                    active_shot = min(shots, key=lambda s: abs(s.peak_frame - i))

                # Modern HUD banner overlay at top
                overlay = ann.copy()
                hud_height = min(110, int(height * 0.22))
                cv2.rectangle(overlay, (0, 0), (width, hud_height), (15, 20, 28), -1)
                cv2.addWeighted(overlay, 0.85, ann, 0.15, 0, ann)

                # Shot ID, Stroke, Direction, Posture
                font_scale = max(0.5, min(0.75, width / 1200.0))
                dir_label = active_shot.direction.upper().replace('_', ' ')
                str_label = active_shot.stroke_type.upper()
                pos_grade_str = active_shot.posture_grade or active_shot.posture.replace('_', ' ')
                pos_label = pos_grade_str.upper()
                pos_score_val = active_shot.posture_score if active_shot.posture_score is not None else int(active_shot.posture_confidence * 100)
                pos_color = (63, 185, 80) if (active_shot.posture_score or 75) >= 70 else (73, 81, 248)

                cv2.putText(ann, f"SHOT {active_shot.shot_id}/{len(shots)}: {str_label} ({active_shot.stroke_confidence:.0%})", 
                            (15, int(hud_height * 0.32)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 220, 150), 2, cv2.LINE_AA)
                cv2.putText(ann, f"DIR: {dir_label} ({active_shot.direction_confidence:.0%})", 
                            (15, int(hud_height * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(ann, f"POSTURE: {pos_label} ({pos_score_val}/100)", 
                            (15, int(hud_height * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, pos_color, 2, cv2.LINE_AA)

                # STROKE peak flash indicator
                is_impact = any(abs(i - s.peak_frame) <= 4 for s in shots)
                if is_impact:
                    cv2.circle(ann, (width - 35, int(hud_height * 0.35)), 12, (0, 0, 255), -1)
                    cv2.putText(ann, "IMPACT", (width - 130, int(hud_height * 0.40)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

                # Frame & Time
                sec = i / fps
                time_str = f"{int(sec//60):02d}:{sec%60:04.1f}"
                cv2.putText(ann, f"{time_str} | Frame {i+1}/{total_frames}", (width - 230, height - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

                out_video.write(ann)

            out_video.release()
            annotated_url = f"/api/videos/{output_filename}"
            logger.info(f"Generated annotated video with {len(shots)} shots: {annotated_url}")
        except Exception as vid_err:
            logger.error(f"Error creating annotated video: {vid_err}")
            annotated_url = None

        return BiomechanicsResponse(
            success=True,
            message=f"Rally analyzed successfully: {len(shots)} distinct stroke(s) detected.",
            is_demo=is_random_fallback,
            total_shots=len(shots),
            shots=shots,
            direction=primary_shot.direction,
            direction_confidence=primary_shot.direction_confidence,
            direction_probabilities=primary_shot.direction_probabilities,
            stroke_type=primary_shot.stroke_type,
            stroke_confidence=primary_shot.stroke_confidence,
            stroke_probabilities=primary_shot.stroke_probabilities,
            posture=primary_shot.posture,
            posture_confidence=primary_shot.posture_confidence,
            posture_probabilities=primary_shot.posture_probabilities,
            recommendation_title=primary_shot.recommendation_title,
            recommendation_text=primary_shot.recommendation_text,
            metrics=primary_shot.metrics,
            pose_history=pose_history,
            frame_metrics=frame_metrics_history,
            annotated_video_url=annotated_url,
            stroke_peak_frame=primary_shot.peak_frame,
            total_frames_analyzed=total_frames
        )
        
    except Exception as e:
        logger.error(f"Error during video processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
