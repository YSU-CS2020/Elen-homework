# Real-Time Human Detection & Tracking

## Description
This project uses YOLOv8 + OpenCV to detect and track multiple humans in real-time using a webcam.

## Features
- Human detection (person only)
- Multi-object tracking with persistent IDs
- Real-time webcam processing
- JSON telemetry output per frame

## Run Instructions

```bash
pip install -r requirements.txt
python main.py


## Output Format

```json
{
  "frame": 1,
  "people": [
    {
      "id": 1,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

## Exit
Press ESC to stop the program.