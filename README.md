---
title: Face Mask AI
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Face Mask AI — Attendance System

Flask web dashboard for face-recognition attendance, using an OpenCV LBPH
recognizer (`cv2.face.LBPHFaceRecognizer`) for recognition. Attendance is
tracked in-memory for the running process (no external spreadsheet
integration).

## Notes

- Lightweight by design: no TensorFlow/DeepFace, so this runs comfortably
  on free-tier hosting (small RAM footprint).
- This Space's filesystem is ephemeral: captured/uploaded face images and
  training artifacts (`face_model.yml`, `labels.json`) are lost on restart
  or rebuild. Re-upload and retrain after each restart if you need a
  populated dataset.
- Camera access (`getUserMedia`) requires HTTPS, which Spaces provides by
  default.
