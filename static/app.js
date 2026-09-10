const camera = document.querySelector('#camera');
const canvas = document.querySelector('#snapshot');
const placeholder = document.querySelector('#camera-placeholder');
const startButton = document.querySelector('#start-camera');
const stopButton = document.querySelector('#stop-camera');
const recognitionState = document.querySelector('#recognition-state');
const recognitionResult = document.querySelector('#recognition-result');
const toast = document.querySelector('#toast');

let stream = null;
let recognitionTimer = null;
let recognitionBusy = false;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  window.setTimeout(() => toast.classList.remove('show'), 3200);
}

function captureFrame() {
  if (!stream || camera.readyState < 2) throw new Error('Camera is not ready');
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  const context = canvas.getContext('2d');
  context.drawImage(camera, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', .82);
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    camera.srcObject = stream;
    placeholder.hidden = true;
    startButton.disabled = true;
    stopButton.disabled = false;
    recognitionState.textContent = 'Đang nhận diện';
    recognitionTimer = window.setInterval(recognize, 1500);
    showToast('Camera đã bật');
  } catch (error) {
    showToast(`Không mở được camera: ${error.message}`);
  }
}

function stopCamera() {
  if (recognitionTimer) window.clearInterval(recognitionTimer);
  recognitionTimer = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  camera.srcObject = null;
  placeholder.hidden = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  recognitionState.textContent = 'Sẵn sàng';
}

async function recognize() {
  if (recognitionBusy || !stream) return;
  recognitionBusy = true;
  try {
    const response = await fetch('/api/recognize', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: captureFrame() })
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error);
    recognitionResult.textContent = data.name === 'Unknown'
      ? 'Không nhận diện được khuôn mặt.'
      : `${data.name}${data.marked ? ' - Đã ghi điểm danh hôm nay.' : ' - Đã nhận diện.'}`;
  } catch (error) {
    recognitionResult.textContent = `Lỗi nhận diện: ${error.message}`;
  } finally {
    recognitionBusy = false;
  }
}

async function captureStage() {
  const name = document.querySelector('#person-name').value.trim();
  const stage = document.querySelector('#capture-stage').value;
  const progress = document.querySelector('#capture-progress');
  if (!name) return showToast('Nhập tên người trước khi chụp');
  if (!stream) return showToast('Hãy bật camera trước');
  const button = document.querySelector('#capture-stage-button');
  button.disabled = true;
  let saved = 0;
  try {
    for (let index = 0; index < 10; index += 1) {
      progress.textContent = `Đang chụp ${index + 1}/10...`;
      const response = await fetch('/api/capture', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, stage, image: captureFrame() })
      });
      const data = await response.json();
      if (data.ok) saved += 1;
      await new Promise(resolve => window.setTimeout(resolve, 350));
    }
    progress.textContent = `Đã lưu ${saved}/10 ảnh cho góc ${stage}.`;
    showToast(`Hoàn tất góc chụp: ${saved}/10 ảnh`);
  } catch (error) {
    progress.textContent = `Lỗi khi chụp: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function uploadDataset() {
  const name = document.querySelector('#upload-name').value.trim();
  const files = document.querySelector('#upload-files').files;
  const output = document.querySelector('#admin-result');
  if (!name || !files.length) return showToast('Nhập tên và chọn ít nhất một ảnh');
  const form = new FormData();
  form.append('name', name);
  Array.from(files).forEach(file => form.append('files', file));
  output.textContent = 'Đang upload và tách khuôn mặt...';
  const response = await fetch('/api/upload', { method: 'POST', body: form });
  const data = await response.json();
  output.textContent = data.ok ? `Đã lưu ${data.saved} ảnh.` : `Upload lỗi: ${data.error}`;
}

async function trainModel() {
  const output = document.querySelector('#admin-result');
  const button = document.querySelector('#train-button');
  button.disabled = true;
  output.textContent = 'Đang train model, có thể mất vài phút...';
  try {
    const response = await fetch('/api/train', { method: 'POST' });
    const data = await response.json();
    output.textContent = data.ok ? 'Train thành công. Xem terminal để biết chi tiết.' : `Train lỗi: ${data.output || data.error}`;
  } catch (error) {
    output.textContent = `Train lỗi: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function checkHealth() {
  const health = document.querySelector('#health');
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    health.textContent = data.credentials_exists ? 'Hệ thống sẵn sàng' : 'Thiếu credentials.json';
  } catch (error) {
    health.textContent = 'Backend chưa sẵn sàng';
  }
}

startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
document.querySelector('#capture-stage-button').addEventListener('click', captureStage);
document.querySelector('#upload-button').addEventListener('click', uploadDataset);
document.querySelector('#train-button').addEventListener('click', trainModel);
checkHealth();
