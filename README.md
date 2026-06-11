# FlowCycle-AI: Real-Time Smart City Traffic Monitor

FlowCycle-AI is a real-time smart city monitoring system that leverages Artificial Intelligence (YOLOv8 & SORT tracking) and IoT (Raspberry Pi) to automatically detect traffic density and dynamically control traffic lights based on a Round-Robin with Queue Priority algorithm.

## Features

- **Real-Time Vehicle Detection:** Uses YOLOv8 for fast and accurate detection of cars, buses, trucks, and motorbikes.
- **Object Tracking:** Employs the SORT (Simple Online and Realtime Tracking) algorithm to count vehicles that pass a designated line, keeping track of moving objects across frames.
- **Dynamic Traffic Light Algorithm:** Utilizes a Round-Robin with Queue Priority algorithm to determine green light times based on queue lengths and weighted vehicle types (e.g., buses and trucks carry higher weight).
- **IoT Integration (Raspberry Pi):** Streams camera feed from a Raspberry Pi via RTSP, and optionally controls physical traffic light LEDs by sending TCP commands to a Raspberry Pi listener script.
- **Gradio User Interface:** Features a clean web UI showing live streams (split into North, East, West, South quadrants), real-time queue states, detection bounding boxes, and comprehensive traffic light stats.

## Project Structure

- `Hardware_code.py`: The main Gradio application that connects to the Raspberry Pi RTSP camera stream, runs YOLO detection + SORT tracking, computes traffic algorithms, and provides the UI.
- `pi_light_controller.py`: The Python socket server intended to run on the Raspberry Pi. It listens for state change signals from the PC and toggles the corresponding physical GPIO LEDs for each direction (Red, Amber, Green).
- `software_code.ipynb`: An offline/development Jupyter notebook for testing the system with pre-recorded videos and performing auto-labeling or ground-truth generation tasks.

## System Requirements

- Python 3.8+
- PyTorch (with CUDA support strongly recommended for real-time processing)
- OpenCV
- Ultralytics (YOLOv8)
- Gradio
- SORT tracking library

## Getting Started

### 1. Raspberry Pi Setup (Camera & Lights)

To stream from the Raspberry Pi and use physical LEDs, deploy the `pi_light_controller.py` on your Pi.

**Run the RTSP Stream on Pi:**
```bash
# Start an RTSP server like mediamtx, then stream:
libcamera-vid -t 0 --inline -o - | \
  ffmpeg -i pipe:0 -c:v libx264 -preset ultrafast \
  -tune zerolatency -f rtsp rtsp://localhost:8554/stream
```

**Run the LED controller:**
```bash
python3 pi_light_controller.py
```

### 2. PC Setup (AI Processor)

Clone the repository and install the dependencies. Make sure your CUDA environment is properly configured.

```bash
git clone https://github.com/kattaabhijeet/FlowCycle-AI.git
cd FlowCycle-AI
pip install -r requirements.txt # (Make sure to install ultralytics, gradio, opencv-python, filterpy, sort)
```

Run the main application:
```bash
python Hardware_code.py
```

1. Enter your Pi's RTSP Stream URL in the UI.
2. Click **Start Monitoring**.
3. Watch the real-time AI processing and see the traffic light phases change dynamically based on congestion!
