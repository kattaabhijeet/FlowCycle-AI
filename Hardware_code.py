import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # Corrected import statement
import gradio as gr
import threading

import time
from PIL import Image
import os
import torch

# ============================================================
# DEVICE CONFIGURATION - Detect and configure GPU ONCE at startup
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_HALF = DEVICE == 'cuda'  # FP16 half-precision only on GPU

print("=" * 50)
print("       DEVICE CONFIGURATION")
print("=" * 50)
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU Name:   {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Ver:   {torch.version.cuda}")
    print(f"Half Prec:  Enabled (FP16)")
else:
    print("WARNING: No CUDA GPU detected! Running on CPU (will be SLOW).")
    print("To install CUDA-enabled PyTorch, run:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
print("=" * 50)

# --- Global variables for the monitoring system ---
model = None
pi_cap = None          # Single VideoCapture for the Raspberry Pi RTSP stream
trackers = {}
vehicle_counts = {}
crossed_vehicles = {}
prev_positions = {}
track_classes = {}     # Stores vehicle type per track ID
monitoring_active = False
current_frame = None
frame_lock = threading.Lock()
monitor_thread = None
stream_status = "Not connected"
# --- Global variables for Traffic Light Control ---
queue_lengths = {"North": 0, "East": 0, "West": 0, "South": 0}
weighted_scores = {"North": 0, "East": 0, "West": 0, "South": 0}  # Density-based weighted time
light_states = {"North": "green", "East": "red", "West": "red", "South": "red"}
green_light_timer = 0
current_green_direction = "North"

# --- Amber phase variables ---
amber_timer = 0
next_green_direction = None
total_light_cycles = 0
green_duration_history = []
start_time = None
served_in_cycle = {"North"}  # North starts as green, so mark it served

# --- Traffic Light Timing Constants ---
MIN_GREEN_TIME = 150   # 5 seconds (in frames at ~30fps)
MAX_GREEN_TIME = 300   # 10 seconds absolute maximum (strict cap for fairness)
MAX_GREEN_TIME_CAP = 300  # 10 seconds - never exceeded
AMBER_DURATION = 60    # 2 seconds

# --- Density-Based Vehicle Weights (seconds per vehicle type) ---
VEHICLE_WEIGHTS = {
    "bus": 5,        # Bus: 5 seconds
    "truck": 5,      # Truck: 5 seconds
    "car": 4,        # Car: 4 seconds
    "motorbike": 3,  # Motorbike: 3 seconds
}

# --- Traffic Light Color Constants ---
LIGHT_COLORS = {
    "red": (0, 0, 255),      # BGR: Red
    "amber": (0, 165, 255),  # BGR: Orange/Amber
    "green": (0, 255, 0),    # BGR: Green
    "dim": (50, 50, 50)      # BGR: Dim gray for inactive lights
}

# --- Vehicle Detection and Tracking Constants ---
vehicle_classes = ["car", "bus", "truck", "motorbike"]
LINE_POSITION_RATIO = 0.50
LINE_THICKNESS = 3
CROSSING_TOLERANCE = 10
DETECTION_BUFFER = 50

def initialize_model():
    """Load YOLOv8s model on GPU with warmup."""
    global model
    try:
        print(f"Initializing model on: {DEVICE}")
        model_options = ["yolov8s.pt", "yolov8n.pt"]
        for model_name in model_options:
            try:
                print(f"Trying to load {model_name}...")
                model = YOLO(model_name)
                print(f"Successfully loaded {model_name}!")
                break
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                if model_name == model_options[-1]:
                    raise Exception("Failed to load any YOLO model.")

        model.to(DEVICE)
        if USE_HALF:
            print("FP16 half-precision enabled for GPU inference!")

        # GPU warmup - pre-compiles CUDA kernels
        print("Warming up GPU...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            model.predict(dummy, conf=0.35, verbose=False, device=DEVICE, half=USE_HALF, imgsz=640)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        print("GPU warm-up complete! Model ready.")
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def split_quad_frame(frame):
    """
    Splits a single Pi quad-view frame (2x2 grid) into 4 directional sub-frames.
    Layout: top-left=North, top-right=East, bottom-left=West, bottom-right=South
    """
    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    return {
        "North": frame[0:mid_h,  0:mid_w].copy(),
        "East":  frame[0:mid_h,  mid_w:w].copy(),
        "West":  frame[mid_h:h,  0:mid_w].copy(),
        "South": frame[mid_h:h,  mid_w:w].copy(),
    }

def setup_stream(stream_url):
    """Connect to the Raspberry Pi RTSP stream and initialise all trackers/counters."""
    global pi_cap, trackers, vehicle_counts, crossed_vehicles, prev_positions
    global track_classes, weighted_scores, queue_lengths, light_states
    global green_light_timer, current_green_direction, amber_timer
    global next_green_direction, total_light_cycles, green_duration_history
    global start_time, served_in_cycle, stream_status

    directions = ["North", "East", "West", "South"]
    try:
        print(f"Connecting to stream: {stream_url}")
        pi_cap = cv2.VideoCapture(stream_url)
        pi_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimise latency

        if not pi_cap.isOpened():
            stream_status = "ERROR: Cannot connect to stream"
            print(stream_status)
            return False

        stream_status = f"Connected: {stream_url}"
        print(stream_status)

        trackers        = {d: Sort(max_age=30, min_hits=2, iou_threshold=0.3) for d in directions}
        vehicle_counts  = {d: 0   for d in directions}
        crossed_vehicles= {d: set() for d in directions}
        prev_positions  = {d: {}  for d in directions}
        track_classes   = {d: {}  for d in directions}
        queue_lengths   = {d: 0   for d in directions}
        weighted_scores = {d: 0   for d in directions}

        light_states            = {"North": "green", "East": "red", "West": "red", "South": "red"}
        green_light_timer       = 0
        current_green_direction = "North"
        amber_timer             = 0
        next_green_direction    = None
        total_light_cycles      = 0
        green_duration_history  = []
        start_time              = time.time()
        served_in_cycle         = {"North"}

        print("Stream setup completed!")
        return True
    except Exception as e:
        stream_status = f"ERROR: {e}"
        print(f"Error setting up stream: {e}")
        return False


def draw_traffic_light(frame, x, y, state, direction_name, timer):
    """
    Draws a realistic 3-light traffic signal (vertical layout)
    - Red light on top
    - Amber/Yellow light in middle
    - Green light on bottom

    Parameters:
    - frame: The image to draw on
    - x, y: top-left corner position
    - state: "red", "amber", or "green"
    - direction_name: "N", "E", "W", "S"
    - timer: current countdown value in seconds
    """
    # Traffic light housing dimensions
    housing_width = 60
    housing_height = 160
    light_radius = 20
    padding = 20

    # Draw black housing background
    cv2.rectangle(frame, (x, y), (x + housing_width, y + housing_height), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + housing_width, y + housing_height), (200, 200, 200), 2)

    # Calculate center positions for each light
    center_x = x + housing_width // 2
    red_y = y + padding + light_radius
    amber_y = y + housing_height // 2
    green_y = y + housing_height - padding - light_radius

    # Determine which lights are lit
    red_color = LIGHT_COLORS["red"] if state == "red" else LIGHT_COLORS["dim"]
    amber_color = LIGHT_COLORS["amber"] if state == "amber" else LIGHT_COLORS["dim"]
    green_color = LIGHT_COLORS["green"] if state == "green" else LIGHT_COLORS["dim"]

    # Draw the three lights
    cv2.circle(frame, (center_x, red_y), light_radius, red_color, -1)
    cv2.circle(frame, (center_x, red_y), light_radius, (100, 100, 100), 2)

    cv2.circle(frame, (center_x, amber_y), light_radius, amber_color, -1)
    cv2.circle(frame, (center_x, amber_y), light_radius, (100, 100, 100), 2)

    cv2.circle(frame, (center_x, green_y), light_radius, green_color, -1)
    cv2.circle(frame, (center_x, green_y), light_radius, (100, 100, 100), 2)

    # Draw direction label
    label_y = y + housing_height + 25
    cv2.putText(frame, direction_name, (center_x - 10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw countdown timer
    if state == "green" or state == "amber":
        timer_text = f"{int(timer)}s"
        timer_color = (0, 255, 255)  # Cyan
    else:
        timer_text = "WAIT"
        timer_color = (100, 100, 100)  # Gray

    timer_y = label_y + 25
    cv2.putText(frame, timer_text, (center_x - 20, timer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

def update_traffic_lights():
    """
    Updates the traffic light states with Round-Robin + Queue Priority.
    Ensures fairness while remaining adaptive to high traffic.
    """
    global green_light_timer, current_green_direction, light_states
    global amber_timer, next_green_direction, total_light_cycles, green_duration_history

    # If we're in amber phase, just count down
    if amber_timer > 0:
        amber_timer += 1
        if amber_timer >= AMBER_DURATION:
            # Amber phase complete - switch to red and activate next green
            light_states[current_green_direction] = "red"
            light_states[next_green_direction] = "green"

            # Record statistics
            total_light_cycles += 1
            green_duration_history.append(green_light_timer / 30.0)  # Convert frames to seconds

            # Update current direction and reset timers
            current_green_direction = next_green_direction
            next_green_direction = None
            amber_timer = 0
            green_light_timer = 0
        return

    # Normal green phase - increment timer
    green_light_timer += 1
    time_to_change = False

    current_queue = queue_lengths[current_green_direction]
    max_other_queue = max([q for d, q in queue_lengths.items() if d != current_green_direction], default=0)

    # Condition 1: Max green time has been exceeded
    if green_light_timer > MAX_GREEN_TIME:
        time_to_change = True
    # Condition 2: Current green lane is empty while others are waiting (after minimum time)
    elif green_light_timer > MIN_GREEN_TIME:
        if current_queue == 0 and sum(queue_lengths.values()) > 0:
            time_to_change = True
    # Condition 3: EMERGENCY OVERRIDE - Another direction has EXTREME congestion
    elif green_light_timer > 60:  # After just 2 seconds (60 frames)
        queue_difference = max_other_queue - current_queue
        # Only switch early if difference is extreme (≥10 vehicles)
        if queue_difference >= 10:
            time_to_change = True

    if time_to_change:
        # Start amber phase
        light_states[current_green_direction] = "amber"
        amber_timer = 1  # Start counting amber phase

        # NEW: Round-Robin with Queue Priority Algorithm
        directions = ["North", "East", "South", "West"]
        current_idx = directions.index(current_green_direction)

        # Default: Next direction in round-robin sequence
        next_in_sequence = directions[(current_idx + 1) % 4]

        # Check if emergency override is needed
        if sum(queue_lengths.values()) == 0:
            # No vehicles anywhere, just follow sequence
            next_green_direction = next_in_sequence
        else:
            # Check for EXTREME congestion (≥10 vehicles difference)
            max_queue = max(queue_lengths.values())
            queue_difference = max_queue - queue_lengths[next_in_sequence]

            if queue_difference >= 10:
                # Emergency: Skip to most congested direction
                next_green_direction = max(queue_lengths, key=queue_lengths.get)
            else:
                # Normal: Follow round-robin sequence (ensures fairness)
                next_green_direction = next_in_sequence

def process_frame():
    """Read one quad-view frame from Pi stream, split into 4 quadrants, detect & track."""
    global current_frame, queue_lengths, weighted_scores

    if pi_cap is None or not pi_cap.isOpened():
        return True  # Signal to stop

    ret, full_frame = pi_cap.read()
    if not ret:
        print("Stream frame dropped or ended.")
        return True

    # Split the single quad-view into 4 directional frames
    quad_frames = split_quad_frame(full_frame)
    directions_order = ["North", "East", "West", "South"]
    processed_frames = []

    for name in directions_order:
        frame = quad_frames[name]

        # Resize to consistent processing width (max 960px)
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))

        frame_height, frame_width = frame.shape[:2]
        counting_line_y = int(frame_height * LINE_POSITION_RATIO)

        # ROI: area above counting line
        roi_frame = frame[0:counting_line_y + DETECTION_BUFFER, :]

        # YOLO inference (GPU optimised)
        with torch.inference_mode():
            results = model.predict(roi_frame, conf=0.35, verbose=False,
                                    device=DEVICE, half=USE_HALF, imgsz=640)

        detections, det_classes = [], []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_name = model.names[int(box.cls[0])]
                if cls_name in vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, float(box.conf[0])])
                    det_classes.append(cls_name)

        detections_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = trackers[name].update(detections_np)
        queue_lengths[name] = len(tracks)

        # IoU class matching for labels
        for tr in tracks:
            tx1, ty1, tx2, ty2, tid = map(int, tr)
            tid = int(tid)
            best_iou, best_cls = 0, None
            for i, det in enumerate(detections):
                dx1, dy1, dx2, dy2 = det[:4]
                ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
                ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (tx2-tx1)*(ty2-ty1) + (dx2-dx1)*(dy2-dy1) - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_cls = iou, det_classes[i]
            if best_cls is not None and best_iou > 0.1:
                track_classes[name][tid] = best_cls

        # Vehicle counting + bounding box drawing
        current_positions = {}
        for tr in tracks:
            x1, y1, x2, y2, track_id = map(int, tr)
            bottom_y = y2
            current_positions[track_id] = bottom_y

            if track_id not in prev_positions[name]:
                prev_positions[name][track_id] = bottom_y
            prev_y = prev_positions[name][track_id]

            if (prev_y < counting_line_y <= bottom_y) and (track_id not in crossed_vehicles[name]):
                vehicle_counts[name] += 1
                crossed_vehicles[name].add(track_id)

            cls_label = track_classes[name].get(track_id, "vehicle")
            box_color = (0, 0, 255) if track_id in crossed_vehicles[name] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{cls_label} {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        prev_positions[name].update(current_positions)

        # Weighted density score
        w_score = sum(VEHICLE_WEIGHTS.get(track_classes[name].get(tid, "car"), 4)
                      for tid in current_positions.keys())
        weighted_scores[name] = w_score

        # Draw counting line
        cv2.line(frame, (0, counting_line_y), (frame_width, counting_line_y), (0, 0, 255), LINE_THICKNESS)

        # Resize to display size
        frame = cv2.resize(frame, (640, 360))

        # Overlay text
        text = f"{name} | Count:{vehicle_counts[name]} Q:{queue_lengths[name]} Wt:{w_score}s"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (5, 5), (15 + tw, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Traffic light
        light_x, light_y = 640 - 90, 360 - 220
        if light_states[name] == "green":
            dyn_g = max(MIN_GREEN_TIME, min(w_score * 30, MAX_GREEN_TIME_CAP))
            timer_value = max(0, (dyn_g - green_light_timer) / 30.0)
        elif light_states[name] == "amber":
            timer_value = max(0, (AMBER_DURATION - amber_timer) / 30.0)
        else:
            timer_value = 0
        draw_traffic_light(frame, light_x, light_y, light_states[name], name[0], timer_value)

        processed_frames.append(frame)

    # Traffic light logic
    update_traffic_lights()

    # Combine into 2x2 grid
    top    = np.hstack((processed_frames[0], processed_frames[1]))  # North, East
    bottom = np.hstack((processed_frames[2], processed_frames[3]))  # West, South
    combined = np.vstack((top, bottom))

    total_counted = sum(vehicle_counts.values())
    if amber_timer > 0:
        status = f"AMBER: {current_green_direction} -> {next_green_direction} | TOTAL: {total_counted}"
        status_color = (0, 165, 255)
    else:
        status = f"GREEN: {current_green_direction} | Time: {green_light_timer/30:.1f}s | TOTAL: {total_counted}"
        status_color = (0, 255, 0)
    (sw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(combined, (5, combined.shape[0]-45), (15+sw, combined.shape[0]-5), (0,0,0), -1)
    cv2.putText(combined, status, (10, combined.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    with frame_lock:
        current_frame = combined
    return False

def monitoring_loop():
    """Main loop: continuously reads Pi stream while active."""
    global monitoring_active
    while monitoring_active:
        try:
            ended = process_frame()
            if ended:
                print("Stream ended or lost.")
                monitoring_active = False
                break
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            monitoring_active = False
            break

def start_monitoring(stream_url):
    """Connect to Pi RTSP stream and start monitoring."""
    global monitoring_active, monitor_thread, current_frame

    stream_url = (stream_url or "").strip()
    if not stream_url:
        return None, "Please enter the Raspberry Pi stream URL."
    if monitoring_active:
        return None, "Monitoring is already active. Stop it first."
    if model is None and not initialize_model():
        return None, "Failed to initialize model. Check logs."
    if not setup_stream(stream_url):
        return None, f"Failed to connect to: {stream_url}"

    monitoring_active = True
    current_frame = None
    monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitor_thread.start()
    return None, f"Connected to Pi stream! Monitoring started."

def stop_monitoring():
    """Stop monitoring and release Pi stream."""
    global monitoring_active, current_frame, pi_cap
    monitoring_active = False
    if monitor_thread:
        monitor_thread.join(timeout=2)
    if pi_cap is not None:
        pi_cap.release()
        pi_cap = None
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    final_counts = dict(vehicle_counts) if vehicle_counts else {}
    print("Monitoring stopped. Final counts:", final_counts)
    return None, f"Stopped. Final counts: {final_counts}"

def get_current_frame():
    """Returns the latest processed frame for Gradio UI."""
    if current_frame is None:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(blank, "No stream active - Enter Pi URL and click Start",
                    (180, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return Image.fromarray(blank)
    with frame_lock:
        frame_rgb = cv2.cvtColor(current_frame.copy(), cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

def get_current_stats():
    """Returns formatted statistics string."""
    if not monitoring_active and not vehicle_counts:
        return "Not monitoring. Enter the Pi stream URL and click Start."

    stats = "=== VEHICLE COUNTS ===\n"
    for d in ["North", "East", "West", "South"]:
        stats += f"{d}: {vehicle_counts.get(d, 0)}\n"
    stats += f"TOTAL: {sum(vehicle_counts.values())}\n\n"

    stats += "=== REAL-TIME QUEUES ===\n"
    for d in ["North", "East", "West", "South"]:
        stats += f"{d}: {queue_lengths.get(d, 0)}\n"

    stats += "\n=== WEIGHTED DENSITY (Green Time) ===\n"
    for d in ["North", "East", "West", "South"]:
        stats += f"{d}: {weighted_scores.get(d, 0)}s\n"
    stats += "(bus/truck=5s, car=4s, bike=3s per vehicle | max 10s)\n\n"

    stats += "=== TRAFFIC LIGHT STATUS ===\n"
    if amber_timer > 0:
        stats += f"PHASE: AMBER (Switching)\n"
        stats += f"From: {current_green_direction}  To: {next_green_direction}\n"
        stats += f"Amber Time: {amber_timer/30:.1f}s\n"
    else:
        dyn_sec = min(max(weighted_scores.get(current_green_direction, 0), 5), 10)
        stats += f"PHASE: GREEN\n"
        stats += f"Current: {current_green_direction}\n"
        stats += f"Green Time: {green_light_timer/30:.1f}s / {dyn_sec}s (weighted)\n"

    stats += "\nLight States:\n"
    for direction, state in light_states.items():
        emoji = "🟢" if state == "green" else ("🟡" if state == "amber" else "🔴")
        stats += f"{direction}: {emoji} {state.upper()}\n"

    if start_time is not None:
        elapsed = time.time() - start_time
        if elapsed > 0:
            stats += "\n=== THROUGHPUT ===\n"
            total_v = sum(vehicle_counts.values())
            stats += f"Total Vehicles: {total_v}\n"
            stats += f"Rate: {(total_v/elapsed)*60:.1f} vehicles/min\n"
            stats += f"Total Cycles: {total_light_cycles}\n"
            if green_duration_history:
                stats += f"Avg Green: {sum(green_duration_history)/len(green_duration_history):.1f}s\n"
    return stats

# ─── Gradio User Interface ────────────────────────────────────────────────────
with gr.Blocks(title="Smart City Traffic Monitor - Raspberry Pi") as demo:
    gr.Markdown("# 🚦 Real-Time Smart City Traffic Monitor")
    gr.Markdown("Connect your Raspberry Pi camera stream. The 2×2 quad-view is automatically split into North / East / West / South and processed independently.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Raspberry Pi Stream")
            stream_url_input = gr.Textbox(
                label="RTSP Stream URL",
                placeholder="tcp://172.20.114.59:8888",
                value="tcp://172.20.114.59:8888",
                info="Enter your Raspberry Pi stream URL"
            )
            connection_status = gr.Textbox(
                label="Connection Status",
                value="Not connected",
                interactive=False
            )

            gr.Markdown("### 2. Control")
            with gr.Row():
                start_btn = gr.Button("▶ Start Monitoring", variant="primary")
                stop_btn  = gr.Button("⏹ Stop Monitoring",  variant="stop")
            status_text = gr.Textbox(label="System Status", interactive=False)

            gr.Markdown("### ℹ️ Raspberry Pi Setup")
            gr.Markdown("""
**Run these commands on your Pi:**
```bash
# 1. Download & start RTSP server (mediamtx)
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_linux_arm64v8.tar.gz
tar -xzf mediamtx*.tar.gz
./mediamtx &

# 2. Stream the Pi camera
libcamera-vid -t 0 --inline -o - | \\
  ffmpeg -i pipe:0 -c:v libx264 -preset ultrafast \\
  -tune zerolatency -f rtsp rtsp://localhost:8554/stream
```
**Quad layout:** top-left = North | top-right = East | bottom-left = West | bottom-right = South
""")

        with gr.Column(scale=3):
            gr.Markdown("### Live Feed")
            live_image = gr.Image(label="Combined Traffic View (4 Directions)", height=600, interactive=False)
            stats_text = gr.Textbox(label="Statistics", lines=20, interactive=False)

    # Event Handlers
    start_btn.click(fn=start_monitoring, inputs=[stream_url_input], outputs=[live_image, status_text])
    stop_btn.click(fn=stop_monitoring,  outputs=[live_image, status_text])

    # Continuous refresh
    timer = gr.Timer(value=0.1, active=True)
    timer.tick(fn=get_current_frame, outputs=[live_image])
    timer.tick(fn=get_current_stats,  outputs=[stats_text])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)

