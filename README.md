# CourtVision AI 🎾 — Tennis Pose & Stroke Analyzer

An advanced, AI-powered tennis biomechanics analysis platform that leverages Computer Vision and Deep Learning to evaluate player technique, detect strokes, and provide kinematic feedback in real time.

---

## 🚀 Key Features

- **Automated Stroke Classification**: Classifies tennis shots (Forehand, Backhand, Serve, Volley, Smash, Slice, etc.) using deep learning models trained on skeletal sequences.
- **MediaPipe Pose Estimation**: Extracts 33 body keypoints across video frames to track kinetic movement without wearable sensors.
- **Biomechanical Angle & Posture Quality**:
  - Calculates elbow angle, shoulder-hip separation, knee flexion, and torso tilt.
  - Scores kinetic chain efficiency and posture balance across preparation, contact, and follow-through phases.
- **Direction & Trajectory Estimation**: Estimates shot direction (Cross-court, Down-the-line, Inside-out) based on hip-shoulder alignment and racquet-arm swing trajectory.
- **Tactical Shot Recommendations**: Provides actionable coaching recommendations tailored to player posture and shot execution.
- **Interactive Web Dashboard**: Modern, glassmorphic UI built with Vanilla JS, HTML5, and CSS featuring live video playback, annotated overlays, and Chart.js metrics.
- **Convenient NPM Workflow**: Launch backend and frontend effortlessly using familiar `npm run dev` and `npm start` commands.

---

## 📁 Repository Structure

```
├── backend/
│   ├── app.py                     # FastAPI application server and endpoints
│   ├── model.py                   # PyTorch neural network architecture & kinematics logic
│   ├── package.json               # Backend npm runner scripts
│   ├── requirements.txt           # Python dependencies
│   ├── test_api.py                # API integration test suite
│   ├── test_bio.py                # Biomechanics calculation tests
│   ├── test_client.py             # Client simulation tests
│   └── test_mp.py                 # MediaPipe pose extraction verification
├── frontend/
│   ├── index.html                 # Analysis dashboard web interface
│   ├── style.css                  # Custom styling and glassmorphism design system
│   ├── app.js                     # Frontend state management & video rendering
│   └── package.json               # Frontend npm runner scripts
├── scripts/
│   ├── start-backend.sh           # Backend launcher (auto-detects .venv)
│   └── start-frontend.sh          # Frontend static HTTP server launcher
├── demo/                          # Sample clips and demonstration assets
├── sample videos/                 # Practice videos of pro and collegiate players
├── outputs/                       # Destination for generated annotated video analyses
├── improved-tennis-pose-extraction-research.ipynb  # Pose extraction & ML experimentation
├── model_random_split.pt          # Pretrained PyTorch model weights
├── package.json                   # Root npm configuration (dev & start scripts)
└── README.md
```

---

## 🛠️ Getting Started

### Prerequisites

- **Node.js & npm** (v16+)
- **Python** (3.9 to 3.11 recommended)
- **pip**
- Modern web browser (Chrome, Safari, Firefox, Edge)

---

### 1. Installation & Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/Prince1941/tennis-player-analysis.git
cd tennis-player-analysis

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install backend Python dependencies
pip install -r backend/requirements.txt
```

---

### 2. Running the Application (Recommended: NPM)

You can launch both services using standard `npm` commands directly from the project root:

#### Start Backend Server
```bash
npm run dev
```
> 🎾 **Backend runs on:** [http://localhost:8000](http://localhost:8000)  
> 📖 **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)  
> *Note: The launcher script automatically detects and uses `.venv/bin/python` without requiring manual virtualenv activation.*

#### Start Frontend Dashboard
Open a new terminal tab and run:
```bash
npm start
```
> 🌐 **Frontend runs on:** [http://localhost:3000](http://localhost:3000)

*(You can also run `npm run dev` from inside `backend/` and `npm start` from inside `frontend/`.)*

---

### 3. Alternative: Running with Python Directly

If you prefer running without `npm`:

```bash
# Terminal 1 — Backend:
source .venv/bin/activate
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend:
python3 -m http.server 3000 --directory frontend
```

---

## 📊 Biomechanical Metrics

| Metric | Description | Target Optimal Range |
| :--- | :--- | :--- |
| **Elbow Angle** | Angle between shoulder, elbow, and wrist during acceleration | Forehand: 120°–160° (semi-straight arm) |
| **Knee Flexion** | Deep knee bend during preparation phase for ground reaction force | 110°–140° |
| **Shoulder-Hip Separation** | Angular difference between shoulder and hip axis (torso coiling) | 25°–45° |
| **Follow-through Balance** | Center-of-mass stability and head alignment at finish | > 85% stability index |

---

## 📄 License

This project is licensed under the MIT License.
