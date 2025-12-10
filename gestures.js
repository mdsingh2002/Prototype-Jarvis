// Hand landmark indices from MediaPipe
const LANDMARKS = {
  WRIST: 0,
  THUMB_TIP: 4,
  INDEX_TIP: 8,
  INDEX_MCP: 5,
  MIDDLE_TIP: 12,
  MIDDLE_MCP: 9,
  RING_TIP: 16,
  RING_MCP: 13,
  PINKY_TIP: 20,
  PINKY_MCP: 17,
};

// Calculate Euclidean distance between two landmarks
function getDistance(landmark1, landmark2) {
  const dx = landmark1.x - landmark2.x;
  const dy = landmark1.y - landmark2.y;
  const dz = landmark1.z - landmark2.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// Check if a finger is extended
function isFingerExtended(landmarks, tipIndex, mcpIndex) {
  const tip = landmarks[tipIndex];
  const mcp = landmarks[mcpIndex];
  const wrist = landmarks[LANDMARKS.WRIST];

  // Distance from tip to wrist vs mcp to wrist
  const tipDistance = getDistance(tip, wrist);
  const mcpDistance = getDistance(mcp, wrist);

  return tipDistance > mcpDistance;
}

// Detect PINCH gesture (thumb and index finger close together)
function detectPinch(landmarks) {
  const thumbTip = landmarks[LANDMARKS.THUMB_TIP];
  const indexTip = landmarks[LANDMARKS.INDEX_TIP];
  const distance = getDistance(thumbTip, indexTip);

  // Threshold for pinch detection (adjust as needed)
  return distance < 0.05;
}

// Detect OPEN PALM (all fingers extended)
function detectOpenPalm(landmarks) {
  const indexExtended = isFingerExtended(
    landmarks,
    LANDMARKS.INDEX_TIP,
    LANDMARKS.INDEX_MCP
  );
  const middleExtended = isFingerExtended(
    landmarks,
    LANDMARKS.MIDDLE_TIP,
    LANDMARKS.MIDDLE_MCP
  );
  const ringExtended = isFingerExtended(
    landmarks,
    LANDMARKS.RING_TIP,
    LANDMARKS.RING_MCP
  );
  const pinkyExtended = isFingerExtended(
    landmarks,
    LANDMARKS.PINKY_TIP,
    LANDMARKS.PINKY_MCP
  );

  return indexExtended && middleExtended && ringExtended && pinkyExtended;
}

// Detect POINTING (only index finger extended)
function detectPointing(landmarks) {
  const indexExtended = isFingerExtended(
    landmarks,
    LANDMARKS.INDEX_TIP,
    LANDMARKS.INDEX_MCP
  );
  const middleExtended = isFingerExtended(
    landmarks,
    LANDMARKS.MIDDLE_TIP,
    LANDMARKS.MIDDLE_MCP
  );
  const ringExtended = isFingerExtended(
    landmarks,
    LANDMARKS.RING_TIP,
    LANDMARKS.RING_MCP
  );
  const pinkyExtended = isFingerExtended(
    landmarks,
    LANDMARKS.PINKY_TIP,
    LANDMARKS.PINKY_MCP
  );

  return indexExtended && !middleExtended && !ringExtended && !pinkyExtended;
}

// Detect FIST (all fingers closed)
function detectFist(landmarks) {
  const indexExtended = isFingerExtended(
    landmarks,
    LANDMARKS.INDEX_TIP,
    LANDMARKS.INDEX_MCP
  );
  const middleExtended = isFingerExtended(
    landmarks,
    LANDMARKS.MIDDLE_TIP,
    LANDMARKS.MIDDLE_MCP
  );
  const ringExtended = isFingerExtended(
    landmarks,
    LANDMARKS.RING_TIP,
    LANDMARKS.RING_MCP
  );
  const pinkyExtended = isFingerExtended(
    landmarks,
    LANDMARKS.PINKY_TIP,
    LANDMARKS.PINKY_MCP
  );

  return !indexExtended && !middleExtended && !ringExtended && !pinkyExtended;
}

// Main gesture detection function
function detectGesture(landmarks) {
  if (detectPinch(landmarks)) {
    return "PINCH";
  } else if (detectPointing(landmarks)) {
    return "POINTING";
  } else if (detectFist(landmarks)) {
    return "FIST";
  } else if (detectOpenPalm(landmarks)) {
    return "OPEN_PALM";
  }
  return "UNKNOWN";
}
