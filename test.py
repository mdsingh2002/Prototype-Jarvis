import cv2
import sys

def test_camera():
    """Test if camera is accessible"""
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot access camera")
        return False

    print("✓ Camera accessible")

    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print("✓ Camera frame captured successfully")
        print(f"  Frame shape: {frame.shape}")
    else:
        print("❌ Error: Cannot read frame from camera")
        cap.release()
        return False

    cap.release()
    return True

def test_imports():
    """Test if all required libraries are installed"""
    print("\nTesting imports...")

    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        return False

    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version: {mp.__version__}")
    except ImportError:
        print("❌ MediaPipe not installed")
        return False

    try:
        import pyautogui
        print(f"✓ PyAutoGUI version: {pyautogui.__version__}")
    except ImportError:
        print("❌ PyAutoGUI not installed")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        return False

    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("Jarvis Hand Tracker - System Test")
    print("=" * 50)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Install missing dependencies with:")
        print("  pip install -r requirements.txt")

    # Test camera
    if not test_camera():
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Ready to run hand tracker.")
        print("\nRun the hand tracker with:")
        print("  python hand_tracker.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("=" * 50)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
