# Jarvis Hand Tracker 🤖

A hand tracking system using OpenCV and MediaPipe that lets you control your computer with hand gestures - like a mini Jarvis! Perfect for browser control, presentations, and even gaming (use at your own account's risk!).

## Features 🚀

- **Real-time hand tracking** using MediaPipe
- **Multiple gesture commands** for computer control
- **Smooth cursor movement** with built-in smoothing algorithm
- **Click, scroll, zoom** and more with simple hand gestures
- **Visual feedback** showing detected hand landmarks and active gestures

## Technologies Used 💻

- **OpenCV**: Camera capture and video processing
- **MediaPipe**: Advanced hand landmark detection
- **PyAutoGUI**: System control (mouse, keyboard)
- **NumPy**: Mathematical operations

## Installation 📦

1. Clone the repository:
```bash
git clone <repository-url>
cd Prototype-Jarvis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test your setup:
```bash
python test.py
```

## Usage 🎯

Run the hand tracker:
```bash
python hand_tracker.py
```

Press **'q'** to quit the application.

## Gesture Commands 👋

| Gesture | Command | Description |
|---------|---------|-------------|
| ☝️ **Index finger up** | Cursor Move | Move your cursor around the screen |
| 🤏 **Thumb + Index pinch** | Left Click | Pinch thumb and index together to click |
| ✌️ **Peace sign (Index + Middle)** | Right Click | Two fingers up for right-click menu |
| ✊ **Closed fist** | Scroll Down | Make a fist to scroll down |
| 🖐️ **Open palm (all fingers up)** | Scroll Up | Show all five fingers to scroll up |
| 🤙 **Thumb + Pinky** | Zoom In | Shaka sign to zoom in (Ctrl+) |
| 🖖 **Three fingers (Index + Middle + Ring)** | Zoom Out | Three fingers to zoom out (Ctrl-) |

## How It Works 🔧

1. **Camera Capture**: OpenCV captures video from your webcam
2. **Hand Detection**: MediaPipe detects hand landmarks (21 points per hand)
3. **Gesture Recognition**: Algorithm analyzes finger positions to identify gestures
4. **Command Execution**: PyAutoGUI executes the corresponding system command
5. **Visual Feedback**: Real-time display shows hand landmarks and active commands

## Project Structure 📁

```
Prototype-Jarvis/
├── hand_tracker.py      # Main hand tracking application
├── test.py              # System test script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Tips for Best Results 💡

- Ensure good lighting for better hand detection
- Keep your hand within camera view
- Use deliberate, clear gestures
- Maintain some distance from the camera (arm's length)
- Reduce background clutter for better tracking

## Future Plans 🔮

- [ ] Voice command integration
- [ ] Multi-hand gesture support
- [ ] Custom gesture mapping
- [ ] Game-specific control profiles
- [ ] Gesture recording and playback
- [ ] Mobile device support

## Troubleshooting 🔍

**Camera not detected:**
- Check camera permissions
- Ensure no other application is using the camera
- Try changing camera index in code (0 to 1 or 2)

**Hand not detected:**
- Improve lighting conditions
- Ensure hand is fully visible in frame
- Adjust `min_detection_confidence` in code

**Laggy performance:**
- Close other applications
- Reduce video resolution
- Lower MediaPipe confidence thresholds

## Safety Note ⚠️

Be cautious when using hand tracking for important tasks or gaming. Accidental gestures may trigger unintended actions. Always test in a safe environment first!

## Contributing 🤝

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new gestures
- Improve detection algorithms
- Add new features

## License 📄

This project is open source and available for educational purposes.

---

**Note:** This is a prototype project. Use responsibly and at your own risk, especially when controlling applications or games!
