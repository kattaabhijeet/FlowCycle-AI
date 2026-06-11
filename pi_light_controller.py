#!/usr/bin/env python3
"""
pi_light_controller.py
=======================
Run this script ON THE RASPBERRY PI.

It starts a TCP socket server that receives traffic light state
commands from the Windows PC (cell6_next.py) and immediately
drives the 4 physical LED traffic light modules via GPIO.

GPIO Pin Mapping (BCM numbering):
  Direction | Red  | Amber | Green
  ----------|------|-------|------
  North     |  17  |  27   |  22
  East      |   5  |   6   |  13
  West      |  19  |  26   |  20
  South     |  16  |  12   |  21

Wiring each module (4-pin LED module):
  Module Pin 1 (GND)   -> Pi GND
  Module Pin 2 (Red)   -> Pi GPIO Red pin (via 330Ω resistor recommended)
  Module Pin 3 (Amber) -> Pi GPIO Amber pin (via 330Ω resistor recommended)
  Module Pin 4 (Green) -> Pi GPIO Green pin (via 330Ω resistor recommended)

Protocol (from Windows PC):
  Plain text, newline-terminated:
  "N:green,E:red,W:red,S:red\n"

Usage:
  python3 pi_light_controller.py
  (Runs on port 9999 by default — ensure firewall allows it)
"""

import socket
import sys
import signal
import time

# ── Try to import RPi.GPIO; fall back to a mock for testing on PC ────────────
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("[GPIO] RPi.GPIO loaded successfully.")
except ImportError:
    GPIO_AVAILABLE = False
    print("[GPIO] RPi.GPIO not found — running in MOCK mode (no physical output).")

# ── Configuration ─────────────────────────────────────────────────────────────
HOST = "0.0.0.0"   # Listen on all interfaces
PORT = 9999         # Must match cell6_next.py PI_LIGHT_PORT

# ── GPIO Pin Mapping (BCM) ────────────────────────────────────────────────────
PIN_MAP = {
    "North": {"red": 17, "amber": 27, "green": 22},
    "East":  {"red":  5, "amber":  6, "green": 13},
    "West":  {"red": 19, "amber": 26, "green": 20},
    "South": {"red": 16, "amber": 12, "green": 21},
}

# Short-name → full direction
SHORT_TO_DIR = {"N": "North", "E": "East", "W": "West", "S": "South"}


def setup_gpio():
    """Initialize GPIO pins as outputs, all OFF."""
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for direction, pins in PIN_MAP.items():
        for color, pin in pins.items():
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    print("[GPIO] All pins initialised (LOW / OFF).")


def set_light(direction: str, state: str):
    """
    Set a single direction's LED module to the given state.
    state: 'red', 'amber', or 'green'
    """
    pins = PIN_MAP.get(direction)
    if pins is None:
        return

    # Determine which LED should be HIGH
    red_on   = (state == "red")
    amber_on = (state == "amber")
    green_on = (state == "green")

    if GPIO_AVAILABLE:
        GPIO.output(pins["red"],   GPIO.HIGH if red_on   else GPIO.LOW)
        GPIO.output(pins["amber"], GPIO.HIGH if amber_on else GPIO.LOW)
        GPIO.output(pins["green"], GPIO.HIGH if green_on else GPIO.LOW)
    else:
        # Mock: just print
        active = "red" if red_on else ("amber" if amber_on else "green")
        print(f"  [MOCK] {direction:6s} → {active.upper()}")


def apply_all_off():
    """Turn every LED off (called on shutdown)."""
    for direction in PIN_MAP:
        set_light(direction, "off")


def parse_and_apply(message: str):
    """
    Parse "N:green,E:red,W:red,S:red" and drive GPIO.
    Silently ignores malformed tokens.
    """
    tokens = message.strip().split(",")
    for token in tokens:
        token = token.strip()
        if ":" not in token:
            continue
        short, state = token.split(":", 1)
        direction = SHORT_TO_DIR.get(short.upper())
        if direction and state in ("red", "amber", "green"):
            set_light(direction, state)


def run_server():
    setup_gpio()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    server_sock.settimeout(1.0)   # Allow Ctrl-C to be caught
    print(f"[Server] Listening on {HOST}:{PORT} …")
    print("[Server] Waiting for connection from Windows PC …\n")

    running = True

    def handle_shutdown(sig, frame):
        nonlocal running
        print("\n[Server] Shutting down …")
        running = False

    signal.signal(signal.SIGINT,  handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while running:
        # ── Accept connection ────────────────────────────────────────────────
        try:
            conn, addr = server_sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        print(f"[Server] Connected: {addr}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle
        buffer = ""

        try:
            while running:
                try:
                    data = conn.recv(256)
                except socket.timeout:
                    continue

                if not data:
                    print(f"[Server] Client {addr} disconnected.")
                    break

                buffer += data.decode("utf-8", errors="ignore")

                # Process all complete lines in the buffer
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        parse_and_apply(line)

        except Exception as e:
            print(f"[Server] Connection error: {e}")
        finally:
            conn.close()

    # ── Cleanup ──────────────────────────────────────────────────────────────
    apply_all_off()
    server_sock.close()
    if GPIO_AVAILABLE:
        GPIO.cleanup()
    print("[Server] GPIO cleaned up. Bye!")


if __name__ == "__main__":
    run_server()
