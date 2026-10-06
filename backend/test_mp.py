import sys
from unittest.mock import MagicMock
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
import mediapipe as mp
print("MediaPipe loaded successfully!", mp.solutions.pose)
