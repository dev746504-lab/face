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

Flask web dashboard for face-recognition attendance, using DeepFace (ArcFace)
for recognition and Google Sheets for attendance logging.

## Configuration

Set the `GOOGLE_CREDENTIALS_JSON` Space secret to the full contents of your
Google service account JSON key (Settings → Repository secrets). The app
reads this env var first and falls back to a local `credentials.json` for
desktop/dev use.

## Notes

- This Space's filesystem is ephemeral: captured/uploaded face images and
  training artifacts are lost on restart or rebuild. Re-upload and retrain
  after each restart if you need a populated dataset.
- DeepFace downloads model weights on first build (see `warmup_models.py`,
  run during the Docker build so the first request isn't slow).
- Camera access (`getUserMedia`) requires HTTPS, which Spaces provides by
  default.
