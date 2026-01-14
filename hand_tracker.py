import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from enum import Enum

class GestureCommand(Enum):
    """Enum for different gesture commands"""
    NONE = 0
    CURSOR_MOVE = 1
    LEFT_CLICK = 2
    RIGHT_CLICK = 3
    SCROLL_UP = 4
    SCROLL_DOWN = 5
    DRAG = 6
    ZOOM_IN = 7
    ZOOM_OUT = 8

class HandTracker:
    def __init__(self):
        """Initialize hand tracking with OpenCV and MediaPipe"""
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # OpenCV setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Smoothing for cursor movement
        self.smoothing = 5
        self.prev_x, self.prev_y = 0, 0

        # Click detection
        self.click_threshold = 40
        self.last_click_time = 0
        self.click_cooldown = 0.5  # seconds

        # Gesture state
        self.is_dragging = False

        print(f"Hand Tracker initialized. Screen size: {self.screen_width}x{self.screen_height}")
        print("\nGesture Commands:")
        print("- Index finger up: Move cursor")
        print("- Index + Thumb pinch: Left click")
        print("- Peace sign (Index + Middle): Right click")
        print("- Closed fist: Scroll down")
        print("- Open palm (all fingers up): Scroll up")
        print("- Thumb + Pinky: Zoom in")
        print("- Three fingers up: Zoom out")
        print("\nPress 'q' to quit")

    def get_finger_state(self, landmarks):
        """Determine which fingers are up"""
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints

        fingers_up = []

        # Thumb (check horizontal distance)
        if landmarks[finger_tips[0]].x < landmarks[finger_pips[0]].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        # Other fingers (check vertical position)
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        return fingers_up

    def get_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def detect_gesture(self, landmarks, fingers_up):
        """Detect which gesture is being performed"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Count fingers up
        total_fingers = sum(fingers_up)

        # Index finger only - Cursor Move
        if fingers_up == [0, 1, 0, 0, 0]:
            return GestureCommand.CURSOR_MOVE, index_tip

        # Thumb + Index pinch - Left Click
        if fingers_up[0] == 1 and fingers_up[1] == 1:
            distance = self.get_distance(thumb_tip, index_tip)
            if distance < 0.05:
                return GestureCommand.LEFT_CLICK, index_tip

        # Peace sign (Index + Middle) - Right Click
        if fingers_up == [0, 1, 1, 0, 0]:
            return GestureCommand.RIGHT_CLICK, index_tip

        # Closed fist - Scroll Down
        if total_fingers == 0 or total_fingers == 1:
            return GestureCommand.SCROLL_DOWN, None

        # All fingers up - Scroll Up
        if total_fingers == 5:
            return GestureCommand.SCROLL_UP, None

        # Thumb + Pinky - Zoom In
        if fingers_up == [1, 0, 0, 0, 1]:
            return GestureCommand.ZOOM_IN, None

        # Three fingers - Zoom Out
        if fingers_up == [0, 1, 1, 1, 0]:
            return GestureCommand.ZOOM_OUT, None

        # Index + Middle + Ring - Drag mode
        if fingers_up == [0, 1, 1, 1, 0] or fingers_up == [1, 1, 1, 1, 0]:
            return GestureCommand.DRAG, index_tip

        return GestureCommand.NONE, None

    def execute_command(self, command, tip_position):
        """Execute the detected gesture command"""
        current_time = time.time()

        if command == GestureCommand.CURSOR_MOVE and tip_position:
            # Convert normalized coordinates to screen coordinates
            x = int(tip_position.x * self.screen_width)
            y = int(tip_position.y * self.screen_height)

            # Apply smoothing
            x = self.prev_x + (x - self.prev_x) // self.smoothing
            y = self.prev_y + (y - self.prev_y) // self.smoothing

            # Move cursor
            pyautogui.moveTo(self.screen_width - x, y)

            self.prev_x, self.prev_y = x, y
            return "Moving Cursor"

        elif command == GestureCommand.LEFT_CLICK:
            if current_time - self.last_click_time > self.click_cooldown:
                pyautogui.click()
                self.last_click_time = current_time
                return "Left Click"

        elif command == GestureCommand.RIGHT_CLICK:
            if current_time - self.last_click_time > self.click_cooldown:
                pyautogui.rightClick()
                self.last_click_time = current_time
                return "Right Click"

        elif command == GestureCommand.SCROLL_DOWN:
            pyautogui.scroll(-10)
            return "Scrolling Down"

        elif command == GestureCommand.SCROLL_UP:
            pyautogui.scroll(10)
            return "Scrolling Up"

        elif command == GestureCommand.ZOOM_IN:
            pyautogui.hotkey('ctrl', '+')
            time.sleep(0.2)
            return "Zooming In"

        elif command == GestureCommand.ZOOM_OUT:
            pyautogui.hotkey('ctrl', '-')
            time.sleep(0.2)
            return "Zooming Out"

        elif command == GestureCommand.DRAG and tip_position:
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True

            # Move while dragging
            x = int(tip_position.x * self.screen_width)
            y = int(tip_position.y * self.screen_height)
            pyautogui.moveTo(self.screen_width - x, y)
            return "Dragging"
        else:
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False

        return "No Command"

    def run(self):
        """Main loop to run hand tracking"""
        print("\nStarting hand tracking...")

        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.hands.process(rgb_frame)

            command_text = "No Hand Detected"

            # If hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # Get finger states
                    fingers_up = self.get_finger_state(hand_landmarks.landmark)

                    # Detect gesture
                    gesture, tip_pos = self.detect_gesture(hand_landmarks.landmark, fingers_up)

                    # Execute command
                    command_text = self.execute_command(gesture, tip_pos)

                    # Display finger states
                    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                    for i, (name, state) in enumerate(zip(finger_names, fingers_up)):
                        cv2.putText(frame, f"{name}: {'Up' if state else 'Down'}",
                                  (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 255, 255), 1)

            # Display command
            cv2.putText(frame, f"Command: {command_text}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Hand Tracking - Jarvis', frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Release resources"""
        print("\nCleaning up...")
        if self.is_dragging:
            pyautogui.mouseUp()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Hand Tracker stopped")

def main():
    """Main entry point"""
    try:
        tracker = HandTracker()
        tracker.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
