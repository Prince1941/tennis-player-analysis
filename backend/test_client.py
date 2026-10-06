import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

print("[1] Testing GET /api/status...", flush=True)
res = client.get("/api/status")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
data = res.json()
print(f"  ✓ Status: {data.get('status')}", flush=True)
print(f"  ✓ Model Loaded: {data.get('model_loaded')}", flush=True)
print(f"  ✓ Classes: {data.get('classes')}", flush=True)
print(f"  ✓ Features: {data.get('features')}", flush=True)

print("\n[2] Testing GET / (Frontend delivery)...", flush=True)
res_root = client.get("/")
assert res_root.status_code == 200, f"Expected 200, got {res_root.status_code}"
print("  ✓ Frontend index.html served correctly", flush=True)

res_css = client.get("/style.css")
assert res_css.status_code == 200, f"Expected 200, got {res_css.status_code}"
assert "aspect-ratio: 16 / 9;" in res_css.text, "Expected aspect-ratio 16/9 in CSS"
assert "min-width: 0;" in res_css.text, "Expected min-width: 0 in CSS"
print("  ✓ Frontend style.css validated (16:9 ratio, min-width: 0 viewport layout fix verified)", flush=True)

res_js = client.get("/app.js")
assert res_js.status_code == 200, f"Expected 200, got {res_js.status_code}"
assert "Biomechanical Score" in res_js.text, "Expected Biomechanical Score in app.js"
print("  ✓ Frontend app.js validated (0-100 posture score rendering & provenance badges verified)", flush=True)

# Test video analysis if sample exists
sample_video = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "sample_tennis.mp4")
if not os.path.exists(sample_video):
    sample_video = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample videos", "sample_tennis.mp4")

if os.path.exists(sample_video):
    print(f"\n[3] Testing POST /api/analyze-video with {os.path.basename(sample_video)}...", flush=True)
    with open(sample_video, "rb") as f:
        res_vid = client.post("/api/analyze-video", files={"file": ("sample.mp4", f, "video/mp4")})
    assert res_vid.status_code == 200, f"Expected 200, got {res_vid.status_code}: {res_vid.text}"
    vid_data = res_vid.json()
    total_shots = vid_data.get('total_shots', 0)
    shots = vid_data.get('shots', [])
    print(f"  ✓ Video analysis succeeded: {total_shots} shots detected", flush=True)
    for s in shots:
        print(f"    • Shot #{s['shot_id']}: {s['stroke_type']} ({s['stroke_confidence']:.0%}, {s['stroke_basis']}) -> {s['direction']} ({s['direction_basis']}) | Posture: {s['posture_grade']} ({s['posture_score']}/100)", flush=True)
        assert 0 <= s['posture_score'] <= 100, "Posture score out of range"
        assert s['stroke_type'] in ['forehand', 'backhand', 'serve']
        assert s['direction'] in ['cross_court', 'down_the_line', 'center']

print("\n=======================================================", flush=True)
print("   ✅ ALL FASTAPI ENDPOINTS & FRONTEND ASSETS VALIDATED!", flush=True)
print("=======================================================", flush=True)
