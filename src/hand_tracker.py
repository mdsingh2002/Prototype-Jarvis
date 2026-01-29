"""
Hand Tracker - Runs in Docker container
Captures webcam feed, detects hand landmarks using MediaPipe,
and sends mouse coordinates to the host via socket.
"""

import os
import socket
import json
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
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

        # Socket connection to host
        self.host = os.environ.get('MOUSE_SERVER_HOST', 'host.docker.internal')
        self.port = int(os.environ.get('MOUSE_SERVER_PORT', 5555))
        self.socket = None

        # Screen resolution (will be received from mouse server)
        self.screen_width = 1920
        self.screen_height = 1080

        # Smoothing buffer for cursor movement
        self.smooth_buffer_size = 5
        self.x_buffer = deque(maxlen=self.smooth_buffer_size)
        self.y_buffer = deque(maxlen=self.smooth_buffer_size)

        # Click detection
        self.pinch_threshold = 0.05  # Distance threshold for pinch detection
        self.last_click_time = 0
        self.click_cooldown = 0.3  # Seconds between clicks

        # Tracking state
        self.tracking_paused = False
        self.palm_hold_frames = 0
        self.palm_hold_threshold = 30  # Frames to hold palm for pause toggle

    def connect_to_server(self):
        """Establish connection to mouse server on host."""
        max_retries = 10
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                print(f"Connected to mouse server at {self.host}:{self.port}")

                # Get screen resolution from server
                data = self.socket.recv(1024).decode('utf-8')
                if data:
                    info = json.loads(data)
                    self.screen_width = info.get('screen_width', 1920)
                    self.screen_height = info.get('screen_height', 1080)
                    print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
                return True
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if self.socket:
                    self.socket.close()
                time.sleep(retry_delay)

        print("Failed to connect to mouse server")
        return False

    def send_command(self, command):
        """Send command to mouse server."""
        if self.socket:
            try:
                message = json.dumps(command) + '\n'
                self.socket.send(message.encode('utf-8'))
            except Exception as e:
                print(f"Error sending command: {e}")
                self.socket = None

    def smooth_coordinates(self, x, y):
        """Apply moving average smoothing to reduce jitter."""
        self.x_buffer.append(x)
        self.y_buffer.append(y)

        smooth_x = sum(self.x_buffer) / len(self.x_buffer)
        smooth_y = sum(self.y_buffer) / len(self.y_buffer)

        return int(smooth_x), int(smooth_y)

    def map_to_screen(self, hand_x, hand_y, frame_width, frame_height):
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

    def process_frame(self, frame):
        """Process a single frame for hand detection."""
        frame_height, frame_width = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

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
                        index_tip.y,
                        frame_width,
                        frame_height
                    )

                    # Apply smoothing
                    smooth_x, smooth_y = self.smooth_coordinates(screen_x, screen_y)

                    # Send move command
                    self.send_command({
                        'action': 'move',
                        'x': smooth_x,
                        'y': smooth_y
                    })

                    # Check for pinch (click)
                    if self.detect_pinch(landmarks):
                        current_time = time.time()
                        if current_time - self.last_click_time > self.click_cooldown:
                            self.send_command({'action': 'click'})
                            self.last_click_time = current_time
                            print("Click!")

        # Draw status on frame
        status_text = "PAUSED" if self.tracking_paused else "TRACKING"
        color = (0, 0, 255) if self.tracking_paused else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame

    def run(self):
        """Main loop for hand tracking."""
        # Connect to mouse server
        if not self.connect_to_server():
            print("Cannot proceed without mouse server connection")
            return

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Hand tracking started. Press 'q' to quit.")
        print("Gestures:")
        print("  - Move index finger to control cursor")
        print("  - Pinch thumb and index finger to click")
        print("  - Hold open palm to pause/resume tracking")

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

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            if self.socket:
                self.socket.close()
            print("Hand tracking stopped.")


if __name__ == '__main__':
    tracker = HandTracker()
    tracker.run()
