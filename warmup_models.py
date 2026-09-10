"""Pre-download DeepFace model weights during build so the first request
after deploy doesn't stall past gunicorn's request timeout."""

import numpy as np
from deepface import DeepFace

dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)

DeepFace.extract_faces(
    img_path=dummy_image,
    detector_backend="retinaface",
    enforce_detection=False,
)
DeepFace.represent(
    img_path=dummy_image,
    model_name="ArcFace",
    enforce_detection=False,
)

print("DeepFace model weights warmed up.")
