const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
const statusElement = document.getElementById("status");
const gestureElement = document.getElementById("gesture");

function updateStatus(message, isActive = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle("active", isActive);
}

function updateGesture(gesture) {
  gestureElement.textContent = `Gesture: ${gesture}`;
  gestureElement.classList.toggle("active", gesture !== "UNKNOWN");
}

function onResults(results) {
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Flip the canvas horizontally to make it easier to understand
  // for the viwer
  canvasCtx.translate(canvasElement.width, 0);
  canvasCtx.scale(-1, 1);

  canvasCtx.drawImage(
    results.image,
    0,
    0,
    canvasElement.width,
    canvasElement.height
  );

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    updateStatus(`Tracking ${results.multiHandLandmarks.length} hand(s)`, true);

    for (const landmarks of results.multiHandLandmarks) {
      // Detect gesture for the first hand
      const gesture = detectGesture(landmarks);
      updateGesture(gesture);

      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5,
      });
      drawLandmarks(canvasCtx, landmarks, {
        color: "#FF0000",
        lineWidth: 2,
        radius: 5,
      });
    }
  } else {
    updateStatus("No hands detected", false);
    updateGesture("NONE");
  }

  canvasCtx.restore();
}

const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  },
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 640,
  height: 480,
});

updateStatus("Starting camera...", false);

camera
  .start()
  .then(() => {
    updateStatus("Camera ready - show your hands!", true);
  })
  .catch((err) => {
    updateStatus(`Error: ${err.message}`, false);
    console.error("Camera error:", err);
  });
