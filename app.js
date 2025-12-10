const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
const statusElement = document.getElementById("status");

function updateStatus(message, isActive = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle("active", isActive);
}

function onResults(results) {
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

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
