"""
Hand Tracker (Local Version) - Runs directly on Windows
Captures webcam feed, detects hand landmarks using MediaPipe,
and controls the mouse directly using pyautogui.

This is a simplified version that runs entirely on the host machine
without requiring Docker.
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


# Safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0  # No delay between actions


class HandTrackerLocal:
    def __init__(self):
        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Screen resolution
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")

        # Smoothing buffer for cursor movement
        self.smooth_buffer_size = 5
        self.x_buffer = deque(maxlen=self.smooth_buffer_size)
        self.y_buffer = deque(maxlen=self.smooth_buffer_size)

        # Click detection
        self.pinch_threshold = 0.05  # Distance threshold for pinch detection
        self.last_click_time = 0
        self.click_cooldown = 0.3  # Seconds between clicks

        # Right click detection (thumb + middle finger)
        self.last_right_click_time = 0

        # Double click detection (peace sign)
        self.last_double_click_time = 0
        self.double_click_cooldown = 0.5

        # Drag mode (fist)
        self.is_dragging = False
        self.fist_threshold = 0.08  # How close fingertips must be to palm

        # Scroll mode
        self.scroll_mode = False
        self.last_scroll_y = None
        self.scroll_sensitivity = 10

        # Tracking state
        self.tracking_paused = False
        self.palm_hold_frames = 0
        self.palm_hold_threshold = 30  # Frames to hold palm for pause toggle

        # Current gesture for display
        self.current_gesture = "NONE"

    def smooth_coordinates(self, x, y):
        """Apply moving average smoothing to reduce jitter."""
        self.x_buffer.append(x)
        self.y_buffer.append(y)

        smooth_x = sum(self.x_buffer) / len(self.x_buffer)
        smooth_y = sum(self.y_buffer) / len(self.y_buffer)

        return int(smooth_x), int(smooth_y)

    def map_to_screen(self, hand_x, hand_y):
        """Map hand position in camera frame to screen coordinates."""
        # Invert x-axis for mirror effect
        hand_x = 1 - hand_x

        # Add margins to make edges easier to reach
        margin = 0.1
        hand_x = (hand_x - margin) / (1 - 2 * margin)
        hand_y = (hand_y - margin) / (1 - 2 * margin)

        # Clamp to valid range
        hand_x = max(0, min(1, hand_x))
        hand_y = max(0, min(1, hand_y))

        # Map to screen
        screen_x = int(hand_x * self.screen_width)
        screen_y = int(hand_y * self.screen_height)

        return screen_x, screen_y

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def detect_pinch(self, landmarks):
        """Detect pinch gesture (thumb tip close to index finger tip)."""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < self.pinch_threshold

    def detect_open_palm(self, landmarks):
        """Detect open palm gesture (all fingers extended)."""
        # Check if all fingertips are above their respective knuckles
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        finger_pips = [6, 10, 14, 18]  # Corresponding PIP joints

        fingers_extended = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers_extended += 1

        # Also check thumb
        if landmarks[4].x < landmarks[3].x:  # Thumb extended (for right hand)
            fingers_extended += 1

        return fingers_extended >= 4

    def detect_fist(self, landmarks):
        """Detect fist gesture (all fingers curled into palm)."""
        # Check if all fingertips are below their respective MCP joints (knuckles)
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        finger_mcps = [5, 9, 13, 17]  # MCP joints (knuckles)

        fingers_curled = 0
        for tip, mcp in zip(finger_tips, finger_mcps):
            if landmarks[tip].y > landmarks[mcp].y:
                fingers_curled += 1

        # Thumb should also be curled (tip close to index base)
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        thumb_curled = self.calculate_distance(thumb_tip, index_mcp) < self.fist_threshold

        return fingers_curled >= 4 and thumb_curled

    def detect_right_click_pinch(self, landmarks):
        """Detect right click gesture (thumb + middle finger pinch)."""
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]

        distance = self.calculate_distance(thumb_tip, middle_tip)
        return distance < self.pinch_threshold

    def detect_peace_sign(self, landmarks):
        """Detect peace sign (index and middle fingers extended, others curled)."""
        # Index and middle should be extended
        index_extended = landmarks[8].y < landmarks[6].y
        middle_extended = landmarks[12].y < landmarks[10].y

        # Ring and pinky should be curled
        ring_curled = landmarks[16].y > landmarks[14].y
        pinky_curled = landmarks[20].y > landmarks[18].y

        return index_extended and middle_extended and ring_curled and pinky_curled

    def detect_scroll_gesture(self, landmarks):
        """Detect scroll gesture (thumb + pinky pinch)."""
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]

        distance = self.calculate_distance(thumb_tip, pinky_tip)
        return distance < self.pinch_threshold * 1.5

    def get_finger_states(self, landmarks):
        """Get which fingers are extended (for debugging/display)."""
        states = []

        # Thumb (different logic - horizontal check)
        if landmarks[4].x < landmarks[3].x:
            states.append("Thumb")

        # Other fingers (vertical check)
        finger_names = ["Index", "Middle", "Ring", "Pinky"]
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for name, tip, pip in zip(finger_names, finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                states.append(name)

        return states

    def process_frame(self, frame):
        """Process a single frame for hand detection."""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            # No hand detected - release drag if active
            if self.is_dragging:
                pyautogui.mouseUp(_pause=False)
                self.is_dragging = False
                print("Drag ended (hand lost)")
            self.current_gesture = "NO HAND"
            self.last_scroll_y = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                landmarks = hand_landmarks.landmark

                # Check for open palm (pause toggle)
                if self.detect_open_palm(landmarks):
                    self.palm_hold_frames += 1
                    if self.palm_hold_frames >= self.palm_hold_threshold:
                        self.tracking_paused = not self.tracking_paused
                        self.palm_hold_frames = 0
                        status = "PAUSED" if self.tracking_paused else "ACTIVE"
                        print(f"Tracking {status}")
                else:
                    self.palm_hold_frames = 0

                if not self.tracking_paused:
                    # Get index finger tip position for cursor
                    index_tip = landmarks[8]

                    # Map to screen coordinates
                    screen_x, screen_y = self.map_to_screen(
                        index_tip.x,
                        index_tip.y
                    )

                    # Apply smoothing
                    smooth_x, smooth_y = self.smooth_coordinates(screen_x, screen_y)

                    current_time = time.time()

                    # Priority-based gesture detection
                    # 1. Fist = Drag mode
                    if self.detect_fist(landmarks):
                        self.current_gesture = "FIST (DRAG)"
                        if not self.is_dragging:
                            pyautogui.mouseDown(_pause=False)
                            self.is_dragging = True
                            print("Drag started")
                        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)

                    # 2. Release drag if fist is released
                    elif self.is_dragging:
                        pyautogui.mouseUp(_pause=False)
                        self.is_dragging = False
                        self.current_gesture = "NONE"
                        print("Drag ended")

                    # 3. Scroll gesture (thumb + pinky)
                    elif self.detect_scroll_gesture(landmarks):
                        self.current_gesture = "SCROLL"
                        if self.last_scroll_y is not None:
                            delta = (self.last_scroll_y - index_tip.y) * self.scroll_sensitivity * 100
                            if abs(delta) > 1:
                                pyautogui.scroll(int(delta), _pause=False)
                        self.last_scroll_y = index_tip.y

                    # 4. Peace sign = Double click
                    elif self.detect_peace_sign(landmarks):
                        self.current_gesture = "PEACE (DBL-CLICK)"
                        if current_time - self.last_double_click_time > self.double_click_cooldown:
                            pyautogui.doubleClick(_pause=False)
                            self.last_double_click_time = current_time
                            print("Double click!")
                        self.last_scroll_y = None

                    # 5. Right click pinch (thumb + middle)
                    elif self.detect_right_click_pinch(landmarks):
                        self.current_gesture = "RIGHT CLICK"
                        if current_time - self.last_right_click_time > self.click_cooldown:
                            pyautogui.rightClick(_pause=False)
                            self.last_right_click_time = current_time
                            print("Right click!")
                        self.last_scroll_y = None

                    # 6. Left click pinch (thumb + index)
                    elif self.detect_pinch(landmarks):
                        self.current_gesture = "LEFT CLICK"
                        if current_time - self.last_click_time > self.click_cooldown:
                            pyautogui.click(_pause=False)
                            self.last_click_time = current_time
                            print("Left click!")
                        self.last_scroll_y = None

                    # 7. Normal cursor movement
                    else:
                        self.current_gesture = "MOVE"
                        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
                        self.last_scroll_y = None

        # Draw status on frame
        status_text = "PAUSED" if self.tracking_paused else "TRACKING"
        color = (0, 0, 255) if self.tracking_paused else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw current gesture
        gesture_color = (255, 255, 0)  # Cyan
        cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)

        # Draw drag indicator
        if self.is_dragging:
            cv2.putText(frame, "DRAGGING", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def run(self):
        """Main loop for hand tracking."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("")
        print("=" * 50)
        print("Hand Tracking Mouse Control")
        print("=" * 50)
        print("")
        print("Gestures:")
        print("  - Index finger up     -> Move cursor")
        print("  - Thumb + Index pinch -> Left click")
        print("  - Thumb + Middle pinch-> Right click")
        print("  - Peace sign (V)      -> Double click")
        print("  - Fist (hold)         -> Drag mode")
        print("  - Thumb + Pinky pinch -> Scroll (move index up/down)")
        print("  - Open palm (hold)    -> Pause/resume tracking")
        print("")
        print("Press 'q' in the camera window to quit")
        print("Move mouse to corner of screen to emergency stop (failsafe)")
        print("")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame")
                    break

                # Process the frame
                frame = self.process_frame(frame)

                # Display the frame
                cv2.imshow('Hand Tracking', frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except pyautogui.FailSafeException:
            print("\nFailsafe triggered! Mouse moved to corner.")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Hand tracking stopped.")


if __name__ == '__main__':
    tracker = HandTrackerLocal()
    tracker.run()
