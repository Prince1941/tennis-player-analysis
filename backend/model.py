import os
import logging
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEQUENCE_LEN = 30  # Fixed temporal sequence length expected by model

# MediaPipe landmark mappings for the 13 relevant joints from notebook Cell 8 & 35
LANDMARK_INDEX_MAP = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

RELEVANT_LANDMARKS = list(LANDMARK_INDEX_MAP.keys())

# Base 39 feature column names (13 landmarks * 3 axes: wx, wy, wz)
FEATURE_COLS = []
for lm in RELEVANT_LANDMARKS:
    for axis in ['wx', 'wy', 'wz']:
        FEATURE_COLS.append(f"{lm}_{axis}")

# Exact label encodings from notebook Cell 34
CLASSES = {
    'direction': ['center', 'cross_court', 'down_the_line'],  # 0: center, 1: cross_court, 2: down_the_line
    'stroke_type': ['backhand', 'forehand'],                   # 0: backhand, 1: forehand
    'posture': ['bad', 'good']                                # 0: bad, 1: good
}

class TennisTransformer(nn.Module):
    """
    Temporal Transformer classifier matching Cell 42 (TennisTransformerGPU).
    Takes a 30-frame sequence of 50 features (39 base + 11 derived).
    Outputs logits for direction (3), stroke_type (2), and posture_quality (2).
    """
    def __init__(self, 
                 input_dim=50,  # 39 base + 11 derived
                 d_model=128,
                 nhead=4,
                 num_layers=4,
                 dropout=0.3,
                 n_directions=3,
                 n_stroke_types=2,
                 n_posture=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQUENCE_LEN, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_directions)
        )
        self.stroke_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_stroke_types)
        )
        self.posture_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_posture)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 30, input_dim)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.direction_head(x), self.stroke_head(x), self.posture_head(x)

def angle_between(a, b, c):
    """Calculates angle (in degrees) at joint b formed by vectors ba and bc."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

def compute_derived_features(base_features):
    """
    Computes 11 biomechanical features from the 39 base landmark coordinates.
    Matches Cell 37 of the research notebook:
      0: right elbow angle
      1: left elbow angle
      2: right shoulder angle
      3: left shoulder angle
      4: right knee angle
      5: left knee angle
      6: shoulder-hip rotation diff (yaw)
      7: trunk lean (nose to hip midpoint angle)
      8: lateral stance width
      9: right wrist rel height
      10: left wrist rel height
    """
    coords = {}
    for name in RELEVANT_LANDMARKS:
        pos = RELEVANT_LANDMARKS.index(name) * 3
        coords[name] = (base_features[pos], base_features[pos+1], base_features[pos+2])

    rs = coords['right_shoulder']
    ls = coords['left_shoulder']
    re = coords['right_elbow']
    le = coords['left_elbow']
    rw = coords['right_wrist']
    lw = coords['left_wrist']
    rh = coords['right_hip']
    lh = coords['left_hip']
    rk = coords['right_knee']
    lk = coords['left_knee']
    ra = coords['right_ankle']
    la = coords['left_ankle']
    nose = coords['nose']

    r_elbow_angle = angle_between(rs, re, rw)
    l_elbow_angle = angle_between(ls, le, lw)
    r_shoulder_angle = angle_between(rh, rs, re)
    l_shoulder_angle = angle_between(lh, ls, le)
    r_knee_angle = angle_between(rh, rk, ra)
    l_knee_angle = angle_between(lh, lk, la)

    # Shoulder-hip rotation difference (yaw)
    shoulder_vec = np.array(rs) - np.array(ls)
    hip_vec = np.array(rh) - np.array(lh)
    shoulder_yaw = np.arctan2(shoulder_vec[2], shoulder_vec[0])
    hip_yaw = np.arctan2(hip_vec[2], hip_vec[0])
    rotation_diff = float(np.degrees(shoulder_yaw - hip_yaw))

    # Trunk lean
    hip_mid = (np.array(rh) + np.array(lh)) / 2.0
    trunk_vec = np.array(nose) - hip_mid
    trunk_lean = float(np.arctan2(trunk_vec[0], trunk_vec[1]))

    # Hip width (lateral stance width)
    hip_width = float(abs(np.array(rh)[0] - np.array(lh)[0]))

    # Wrist height relative to shoulder midpoint
    shoulder_mid_y = (np.array(rs)[1] + np.array(ls)[1]) / 2.0
    r_wrist_rel_height = float(np.array(rw)[1] - shoulder_mid_y)
    l_wrist_rel_height = float(np.array(lw)[1] - shoulder_mid_y)

    return [
        r_elbow_angle, l_elbow_angle,
        r_shoulder_angle, l_shoulder_angle,
        r_knee_angle, l_knee_angle,
        rotation_diff,
        trunk_lean,
        hip_width,
        r_wrist_rel_height,
        l_wrist_rel_height
    ]

def normalize_and_resample_sequence(base_seq, seq_len=SEQUENCE_LEN):
    """
    Featurizes and normalizes a sequence of base landmark frames matching Cell 38:
    1. Computes 11 derived features per frame.
    2. Z-score normalizes base features across the sequence: (b - mean) / (std + 1e-8).
    3. Z-score normalizes derived features across the sequence: (d - mean) / (std + 1e-8).
    4. Concatenates into 50-dimensional feature vectors.
    5. Resamples (via downsampling or linear interpolation) to exact seq_len (30).
    """
    base_arr = np.array(base_seq, dtype=np.float32)
    n_frames = len(base_arr)
    if n_frames == 0:
        return np.zeros((seq_len, 50), dtype=np.float32), np.zeros((seq_len, 11), dtype=np.float32)

    # Compute un-normalized derived features for biomechanical metrics
    derived_list = [compute_derived_features(base_arr[i]) for i in range(n_frames)]
    derived_arr = np.array(derived_list, dtype=np.float32)

    # Per-window Z-score normalization matching notebook Cell 38
    b_mean = base_arr.mean(axis=0)
    b_std = base_arr.std(axis=0) + 1e-8
    b_norm = (base_arr - b_mean) / b_std

    d_mean = derived_arr.mean(axis=0)
    d_std = derived_arr.std(axis=0) + 1e-8
    d_norm = (derived_arr - d_mean) / d_std

    combined = np.concatenate([b_norm, d_norm], axis=1)  # (n_frames, 50)

    # Resample to fixed seq_len (30)
    total_dim = combined.shape[1]
    if n_frames == seq_len:
        resampled_combined = combined
        resampled_derived = derived_arr
    elif n_frames > seq_len:
        indices = np.linspace(0, n_frames - 1, seq_len).astype(int)
        resampled_combined = combined[indices]
        resampled_derived = derived_arr[indices]
    else:
        x_old = np.linspace(0, 1, n_frames)
        x_new = np.linspace(0, 1, seq_len)
        resampled_combined = np.zeros((seq_len, total_dim), dtype=np.float32)
        for j in range(total_dim):
            resampled_combined[:, j] = np.interp(x_new, x_old, combined[:, j])
            
        resampled_derived = np.zeros((seq_len, 11), dtype=np.float32)
        for j in range(11):
            resampled_derived[:, j] = np.interp(x_new, x_old, derived_arr[:, j])

    return resampled_combined, resampled_derived

def classify_stroke_kinematic_consensus(stroke_base_window, str_probs):
    """
    Kinematic consensus stroke classification resolving the trained model's
    Backhand bias caused by preparation-window truncation in the training dataset.

    Evaluates:
      1. Overhead wrist elevation relative to nose -> Serve/Smash detection.
      2. Dominant hitting arm lateral displacement relative to the transverse shoulder axis.
      3. Consensus weighting between 3D kinematics (65%) and PyTorch model logits (35%).
    """
    base_arr = np.array(stroke_base_window, dtype=np.float32)
    n_frames = len(base_arr)
    if n_frames < 3:
        # Fallback to model probabilities if insufficient frames
        idx = int(np.argmax(str_probs))
        stroke_type = CLASSES['stroke_type'][idx]
        conf = float(str_probs[idx])
        prob_map = {CLASSES['stroke_type'][i]: round(float(str_probs[i]), 3) for i in range(len(CLASSES['stroke_type']))}
        return stroke_type, conf, "PyTorch Model", prob_map

    def get_joint(frame_idx, joint_name):
        pos = RELEVANT_LANDMARKS.index(joint_name) * 3
        return base_arr[frame_idx, pos:pos+3]

    # 1. Overhead Serve / Smash Detection
    # In MediaPipe world coordinates, Y is negative upwards towards the sky
    nose_y = [get_joint(i, 'nose')[1] for i in range(n_frames)]
    rw_y = [get_joint(i, 'right_wrist')[1] for i in range(n_frames)]
    lw_y = [get_joint(i, 'left_wrist')[1] for i in range(n_frames)]
    min_wrist_y = min(min(rw_y), min(lw_y))
    min_nose_y = min(nose_y)

    if min_wrist_y < (min_nose_y - 0.22):
        # Wrist extends > 22cm above head apex -> Serve / Overhead Smash
        return "serve", 0.94, "Kinematic Overhead Trajectory Analysis", {
            "forehand": 0.03,
            "backhand": 0.03,
            "serve": 0.94
        }

    # 2. Groundstroke Forehand vs. Backhand Lateral Kinematic Analysis
    # Evaluates dominant right wrist position relative to the left-to-right shoulder vector
    lateral_projections = []
    for i in range(n_frames):
        ls = get_joint(i, 'left_shoulder')
        rs = get_joint(i, 'right_shoulder')
        rw = get_joint(i, 'right_wrist')

        shoulder_vec = rs[:2] - ls[:2]
        shoulder_width = np.linalg.norm(shoulder_vec) + 1e-8
        shoulder_mid = (rs[:2] + ls[:2]) / 2.0
        wrist_rel = rw[:2] - shoulder_mid

        # Normalized projection: > 0 means dominant right (forehand) side of torso
        proj = float(np.dot(wrist_rel, shoulder_vec) / (shoulder_width ** 2))
        lateral_projections.append(proj)

    lateral_projections = np.array(lateral_projections)
    prep_proj = float(np.mean(lateral_projections[:int(n_frames * 0.45)]))
    contact_proj = float(np.mean(lateral_projections[int(n_frames * 0.45):int(n_frames * 0.75)]))
    follow_proj = float(np.mean(lateral_projections[int(n_frames * 0.75):]))

    # Forehand kinematics: racquet prepared and contacted on the dominant lateral side (proj > 0.05)
    # Backhand kinematics: racquet drawn across the torso (proj < -0.15) and uncoiled forward
    if prep_proj > -0.05 and contact_proj > 0.05:
        kin_fh_score = 0.88
    elif prep_proj < -0.15 or (contact_proj < -0.05 and follow_proj > prep_proj):
        kin_fh_score = 0.12
    else:
        kin_fh_score = float(np.clip(0.50 + contact_proj, 0.10, 0.90))

    model_fh_prob = float(str_probs[1]) # 'forehand' is index 1
    model_bh_prob = float(str_probs[0]) # 'backhand' is index 0

    # Consensus blending: 65% kinematic trajectory evidence, 35% model logit evidence
    blended_fh = float(0.65 * kin_fh_score + 0.35 * model_fh_prob)
    blended_bh = 1.0 - blended_fh

    if blended_fh >= 0.50:
        stroke_type = "forehand"
        stroke_conf = round(blended_fh, 3)
    else:
        stroke_type = "backhand"
        stroke_conf = round(blended_bh, 3)

    if (model_fh_prob >= 0.60 and blended_fh >= 0.60) or (model_bh_prob >= 0.60 and blended_bh >= 0.60):
        stroke_basis = "PyTorch Transformer Consensus"
    else:
        stroke_basis = "Kinematic Trajectory Consensus (Corrected Training Bias)"

    prob_map = {
        "backhand": round(blended_bh, 3),
        "forehand": round(blended_fh, 3)
    }

    return stroke_type, stroke_conf, stroke_basis, prob_map

def estimate_shot_direction(stroke_base_window, dir_probs, stroke_type):
    """
    Modular shot direction classifier incorporating follow-through vector geometry
    and hip-shoulder yaw rotation to overcome the 97% 'center' training dataset imbalance.

    Returns transparent provenance indicating kinematic vector estimation.
    """
    base_arr = np.array(stroke_base_window, dtype=np.float32)
    n_frames = len(base_arr)
    if n_frames < 4:
        idx = int(np.argmax(dir_probs))
        return CLASSES['direction'][idx], float(dir_probs[idx]), "PyTorch Model", {
            CLASSES['direction'][i]: round(float(dir_probs[i]), 3) for i in range(len(CLASSES['direction']))
        }

    def get_joint(frame_idx, joint_name):
        pos = RELEVANT_LANDMARKS.index(joint_name) * 3
        return base_arr[frame_idx, pos:pos+3]

    # Evaluate follow-through vector from contact (~50%) to finish (~85%)
    idx_contact = int(n_frames * 0.50)
    idx_finish = min(n_frames - 1, int(n_frames * 0.85))

    rw_contact = get_joint(idx_contact, 'right_wrist')
    rw_finish = get_joint(idx_finish, 'right_wrist')
    delta_x = float(rw_finish[0] - rw_contact[0])
    delta_z = float(rw_finish[2] - rw_contact[2])
    follow_angle = float(abs(np.degrees(np.arctan2(delta_x, delta_z + 1e-6))))

    # Shoulder-hip rotation diff at contact
    rs = get_joint(idx_contact, 'right_shoulder')
    ls = get_joint(idx_contact, 'left_shoulder')
    rh = get_joint(idx_contact, 'right_hip')
    lh = get_joint(idx_contact, 'left_hip')
    sh_yaw = float(np.arctan2(rs[2] - ls[2], rs[0] - ls[0]))
    hp_yaw = float(np.arctan2(rh[2] - lh[2], rh[0] - lh[0]))
    rot_diff = float(abs(np.degrees(sh_yaw - hp_yaw)))

    # Direction categorization based on angular follow-through and rotation
    if follow_angle > 22.0 or rot_diff > 30.0:
        kin_dir = "cross_court"
        kin_probs = {"cross_court": 0.70, "center": 0.18, "down_the_line": 0.12}
    elif follow_angle < 12.0 and rot_diff < 22.0:
        kin_dir = "down_the_line"
        kin_probs = {"cross_court": 0.12, "center": 0.20, "down_the_line": 0.68}
    else:
        kin_dir = "center"
        kin_probs = {"cross_court": 0.22, "center": 0.62, "down_the_line": 0.16}

    # Model probabilities from checkpoint
    model_probs = {CLASSES['direction'][i]: float(dir_probs[i]) for i in range(len(CLASSES['direction']))}

    # Blended probabilities: 70% kinematic trajectory (direction is kinetic follow-through), 30% model
    blended_probs = {}
    for d in CLASSES['direction']:
        blended_probs[d] = round(0.70 * kin_probs[d] + 0.30 * model_probs.get(d, 0.33), 3)

    best_dir = max(blended_probs, key=blended_probs.get)
    best_conf = float(blended_probs[best_dir])

    return best_dir, best_conf, "Kinematic Follow-Through Vector (Estimated)", blended_probs

def evaluate_posture_quality(derived_history, stroke_type, pos_probs):
    """
    Evidence-based multi-factor posture quality evaluation.
    Computes an objective 0–100 biomechanical score across 4 key pillars (25 pts each):
      1. Knee Flexion / Ground Loading (0–25 pts)
      2. Trunk Rotation & Coiling Torque (0–25 pts)
      3. Core Stability & Dynamic Lean (0–25 pts)
      4. Arm Extension & Kinetic Lever (0–25 pts)

    Produces genuine score differentiation (Excellent, Good, Needs Improvement, Poor)
    and pinpoints specific kinetic flaws.
    """
    derived_arr = np.array(derived_history, dtype=np.float32)
    n_frames = len(derived_arr)
    if n_frames == 0:
        return {
            "score": 75,
            "grade": "Good",
            "confidence": 0.75,
            "flaws": [],
            "sub_scores": {"knee_loading": 20, "trunk_rotation": 20, "core_balance": 20, "arm_extension": 15}
        }

    # Extract metrics across the stroke
    r_knee_angles = derived_arr[:, 4]
    rot_diffs = derived_arr[:, 6]
    trunk_leans = np.abs(np.degrees(derived_arr[:, 7])) # Converted strictly to degrees!
    r_elbow_angles = derived_arr[:, 0]

    min_knee = float(np.min(r_knee_angles))
    max_knee = float(np.max(r_knee_angles))
    avg_rot = float(np.mean(rot_diffs))
    max_rot = float(np.max(rot_diffs))
    avg_lean = float(np.mean(trunk_leans))
    avg_elbow = float(np.mean(r_elbow_angles))

    flaws = []

    # 1. Knee Loading (0-25 pts) - Athletic ground force preparation
    if 110.0 <= min_knee <= 145.0:
        s_knee = 25
    elif (146.0 <= min_knee <= 158.0) or (98.0 <= min_knee <= 109.0):
        s_knee = 18
    elif min_knee > 158.0:
        s_knee = 10
        flaws.append(f"Upright knee angle ({min_knee:.1f}°) limits lower-body kinetic loading")
    else:
        s_knee = 10
        flaws.append(f"Excessive knee flexion collapse ({min_knee:.1f}°), risking uncentered contact")

    # 2. Trunk Rotation (0-25 pts) - Shoulder-hip coiling torque
    if max_rot >= 26.0:
        s_rot = 25
    elif max_rot >= 18.0:
        s_rot = 18
    elif max_rot >= 10.0:
        s_rot = 12
        flaws.append(f"Insufficient trunk coiling ({max_rot:.1f}°), restricting rotational power transfer")
    else:
        s_rot = 6
        flaws.append(f"Torso uncoiled ({max_rot:.1f}°), resulting in an arm-dominant stroke")

    # 3. Core Balance & Lean (0-25 pts) - Center of gravity control (in degrees)
    if avg_lean <= 18.0:
        s_bal = 25
    elif avg_lean <= 28.0:
        s_bal = 18
    elif avg_lean <= 38.0:
        s_bal = 10
        flaws.append(f"Torso lean ({avg_lean:.1f}°) destabilizes dynamic balance at contact")
    else:
        s_bal = 5
        flaws.append(f"Severe lateral torso lean ({avg_lean:.1f}°), throwing head off contact plane")

    # 4. Arm Extension (0-25 pts) - Fluid kinetic lever
    if 105.0 <= avg_elbow <= 155.0:
        s_arm = 25
    elif (90.0 <= avg_elbow <= 104.0) or (156.0 <= avg_elbow <= 168.0):
        s_arm = 18
    elif avg_elbow < 90.0:
        s_arm = 10
        flaws.append(f"Cramped elbow angle ({avg_elbow:.1f}°), contact is jammed too close to body")
    else:
        s_arm = 10
        flaws.append(f"Over-extended elbow ({avg_elbow:.1f}°), risking tendon strain on impact")

    total_score = s_knee + s_rot + s_bal + s_arm

    if total_score >= 82:
        grade = "Excellent"
    elif total_score >= 68:
        grade = "Good"
    elif total_score >= 50:
        grade = "Needs Improvement"
    else:
        grade = "Poor"

    return {
        "score": int(total_score),
        "grade": grade,
        "confidence": round(float(total_score / 100.0), 2),
        "flaws": flaws,
        "sub_scores": {
            "knee_loading": s_knee,
            "trunk_rotation": s_rot,
            "core_balance": s_bal,
            "arm_extension": s_arm
        }
    }

def recommend_shot(direction, stroke_type, posture, metrics=None, posture_eval=None):
    """
    Dynamic, personalized coaching recommendation engine.
    Integrates actual joint measurements in degrees (°) against ATP/ITF reference standards,
    diagnoses kinetic deficits, and provides specific corrective drills.
    """
    direction_str = direction.lower()
    stroke_str = stroke_type.lower()
    posture_str = posture.lower()

    # Telemetry extracted strictly in degrees
    knee_angle = metrics.get('max_right_knee_angle', metrics.get('right_knee_angle', 138.0)) if metrics else 138.0
    trunk_rot = metrics.get('max_trunk_rotation_diff', metrics.get('trunk_rotation_diff', 29.0)) if metrics else 29.0
    # Ensure trunk lean is in degrees (no radians displayed!)
    raw_lean = metrics.get('avg_trunk_lean', metrics.get('trunk_lean', 12.0)) if metrics else 12.0
    trunk_lean_deg = abs(raw_lean) if abs(raw_lean) > 1.0 else abs(np.degrees(raw_lean))
    elbow_angle = metrics.get('avg_right_elbow_angle', metrics.get('right_elbow_angle', 124.0)) if metrics else 124.0

    posture_grade = posture_eval.get("grade", "Good") if posture_eval else posture.capitalize()
    posture_score = posture_eval.get("score", 75) if posture_eval else 75
    flaws = posture_eval.get("flaws", []) if posture_eval else []

    title = f"{stroke_str.title()} ({direction_str.title().replace('_', ' ')}): {posture_grade} Mechanics ({posture_score}/100)"

    paragraphs = []
    paragraphs.append(
        f"<b>Tactical Assessment:</b> {direction_str.title().replace('_', ' ')} {stroke_str.title()} "
        f"— Evaluated with an overall biomechanical posture score of <b>{posture_score}/100</b> ({posture_grade})."
    )

    # Telemetry vs Reference Ranges (strictly in degrees)
    paragraphs.append(
        f"<b>Observed Telemetry vs. ATP Reference Ranges:</b><br>"
        f"• <i>Trunk Rotation Coil:</i> <b>{trunk_rot:.1f}°</b> (Pro Reference: 28°–45° shoulder-hip separation)<br>"
        f"• <i>Knee Flexion Loading:</i> <b>{knee_angle:.1f}°</b> (Pro Reference: 110°–140° athletic loading)<br>"
        f"• <i>Core Torso Tilt:</i> <b>{trunk_lean_deg:.1f}°</b> (Pro Reference: 6°–20° dynamic lean)<br>"
        f"• <i>Elbow Extension:</i> <b>{elbow_angle:.1f}°</b> (Pro Reference: 110°–150° semi-western contact)"
    )

    if flaws:
        flaw_items = "".join([f"• ⚠️ {f}<br>" for f in flaws])
        paragraphs.append(f"<b>Identified Kinetic Deficits:</b><br>{flaw_items}")

    # Actionable Corrective Drills
    if posture_score >= 80:
        drills = (
            "1. <b>Rhythm & Recovery Footwork:</b> Maintain active split-step recovery immediately after follow-through.<br>"
            "2. <b>Depth Consistency:</b> Focus on hitting 1 meter past the service line with heavy topspin clearance."
        )
    else:
        drills = (
            "1. <b>Unit Turn Shoulder Coil Drill:</b> Initiate the backswing with both hands on the racquet to guarantee 30°+ torso rotation.<br>"
            "2. <b>Drop & Drive Footwork:</b> Deepen knee flexion to ~125° during loading to generate kinetic ground force into impact.<br>"
            "3. <b>Head-Still Contact Hold:</b> Keep eyes fixed on contact point through extension to prevent upper torso drift."
        )

    paragraphs.append(f"<b>Prescribed Tennis Drills:</b><br>{drills}")

    return title, "<br><br>".join(paragraphs)

def get_model(model_path="model_random_split.pt", force_cpu=True):
    """
    Loads TennisTransformer model checkpoint.
    Ensures input_dim matches checkpoint (50) and model is in eval mode.
    """
    if force_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    input_dim = 50
    state_dict = None
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            if 'input_proj.weight' in state_dict:
                input_dim = state_dict['input_proj.weight'].shape[1]
                logger.info(f"Loaded checkpoint from {model_path} with input_dim={input_dim}.")
        except Exception as e:
            logger.error(f"Error reading model weights from {model_path}: {e}")

    model = TennisTransformer(input_dim=input_dim)
    
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info(f"Successfully loaded trained weights from {model_path} onto {device}")
            return model, False, device
        except Exception as e:
            logger.error(f"Failed to load state_dict: {e}. Falling back to randomized weights.")
    
    logger.warning(f"Using randomized fallback model weights.")
    model.to(device)
    model.eval()
    try:
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        logger.error(f"Could not save fallback weights: {e}")
        
    return model, True, device
