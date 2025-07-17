# Accurate Shake Test

Anti-shake Vibration Test Program - Detects crosshair vibration amplitude in videos

## Features

- **Precise Detection**: Uses multiple detection methods (template matching, line detection, contour detection) to accurately locate crosshair center
- **Real-time Analysis**: Real-time tracking of crosshair position changes, displaying vibration trajectory and extreme points
- **Multi-dimensional Measurement**: Separately measures vibration range in X and Y directions, calculates maximum vibration value
- **Interactive ROI Selection**: Supports manual selection of region of interest to improve detection accuracy
- **Global Keyboard Listener**: Supports global hotkey control without window focus
- **Test Report**: Automatically generates detailed test reports, supports batch testing of multiple videos

## System Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Windows system (for global keyboard listener)

## Installation

```bash
pip install opencv-python numpy
```

## Usage

1. **Run the Program**
   ```bash
   python accurate_shake_test.py
   ```

2. **Select Video File**
   - Program automatically scans video files in current directory and subdirectories
   - Use up/down arrow keys to select video file
   - Press Enter to confirm selection

3. **ROI Selection**
   - Drag mouse on displayed video frame to select region containing crosshair
   - Ensure selected region is larger than 30x30 pixels
   - Analysis will start automatically after selection

4. **Playback Controls**
   - `q` or `ESC`: Exit program
   - `p`: Pause/resume playback
   - `+`: Speed up playback
   - `-`: Slow down playback
   - `Space`: Single frame step (when paused)

## Test Standards

- **Pass Criteria**: Vibration amplitude ≤ 10 pixels is considered pass
- **Measurement Method**: Calculate difference between maximum and minimum values in X and Y directions separately
- **Final Result**: Take maximum difference between X and Y directions as maximum vibration value

## Output Results

Program generates `测试报告.txt` file containing:
- Detailed test results for each video
- Extreme point coordinates in X and Y directions
- Vibration amplitude calculation process
- Test pass/fail status
- Test statistical summary

## Detection Principles

The program uses a combination of three detection methods:

1. **Template Matching**: Template matching based on crosshair shape
2. **Line Detection**: Detects line intersections using Hough transform
3. **Contour Detection**: Contour centroid detection based on morphological operations

Detection results consider both confidence and position stability to select optimal detection results.

## Notes

- Program rotates input video 90 degrees clockwise for display
- Records raw position data during detection without filtering or smoothing
- Supports multiple video formats: MP4, AVI, MOV, MKV
- Recommended for use with videos under good lighting conditions

## Author

Anti-shake Vibration Test Program v1.0

## License

This project is for learning and research purposes only.