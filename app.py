import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import threading
import queue
import time
import logging
import json
from datetime import datetime
import geocoder
import os

# Set up logging p




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load YOLO model
try:
    model = YOLO('best.pt')
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    model = None

# Global variables
frame_queue = queue.Queue(maxsize=5)
detection_results = {
    "potholes": 0,
    "status": "stopped",
    "message": "Ready to start",
    "current_location": None
}
camera = None
is_running = False
latest_frame = None

# Store detected potholes
detected_potholes = []
MAX_RECENT_DETECTIONS = 50

# Data file for persistent storage
DATA_FILE = "pothole_data.json"


def is_render():
    """Check if running on Render"""
    return 'RENDER' in os.environ


def load_pothole_data():
    """Load pothole data from JSON file"""
    global detected_potholes
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                detected_potholes = json.load(f)
            logger.info(f"Loaded {len(detected_potholes)} pothole records")
    except Exception as e:
        logger.error(f"Error loading pothole data: {e}")
        detected_potholes = []


def save_pothole_data():
    """Save pothole data to JSON file"""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(detected_potholes, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving pothole data: {e}")


def get_current_location():
    """Get current geolocation using geocoder"""
    if is_render():
        # On Render, return mock location since we can't access camera anyway
        return {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "address": "Render Cloud Server",
            "city": "New York",
            "country": "USA",
            "timestamp": datetime.now().isoformat(),
            "note": "Mock location - Camera not available on cloud"
        }

    try:
        g = geocoder.ip('me')
        if g.ok:
            return {
                "latitude": g.latlng[0],
                "longitude": g.latlng[1],
                "address": g.address,
                "city": g.city,
                "country": g.country,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting location: {e}")

    return None


def create_placeholder_frame(message="Camera not available"):
    """Create a placeholder frame"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    cv2.putText(frame, message, (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def generate_frames():
    """Generate frames with bounding boxes for video streaming"""
    global latest_frame

    while True:
        try:
            if latest_frame is not None:
                frame_data = latest_frame
            else:
                frame_data = create_placeholder_frame("No video feed")

            ret, buffer = cv2.imencode('.jpg', frame_data)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.033)
        except Exception as e:
            logger.error(f"Error in generate_frames: {e}")
            time.sleep(0.1)


def add_pothole_detection(location, confidence=0.0):
    """Add a new pothole detection to the list"""
    global detected_potholes

    detection = {
        "id": len(detected_potholes) + 1,
        "timestamp": datetime.now().isoformat(),
        "location": location,
        "confidence": confidence,
        "latitude": location.get("latitude", 0),
        "longitude": location.get("longitude", 0),
        "address": location.get("address", "Unknown"),
        "city": location.get("city", "Unknown")
    }

    is_new = True
    if detected_potholes:
        latest = detected_potholes[0]
        time_diff = (datetime.now() - datetime.fromisoformat(latest["timestamp"])).total_seconds()
        if time_diff < 60 and location.get("latitude") == latest["latitude"]:
            is_new = False

    if is_new:
        detected_potholes.insert(0, detection)
        if len(detected_potholes) > MAX_RECENT_DETECTIONS:
            detected_potholes = detected_potholes[:MAX_RECENT_DETECTIONS]

        save_pothole_data()
        logger.info(f"New pothole detected at {location.get('address', 'Unknown location')}")

    return is_new


def run_detection():
    """Run pothole detection on camera feed"""
    global camera, is_running, detection_results, latest_frame

    if is_render():
        # On Render, simulate detection since camera isn't available
        detection_results["status"] = "simulated"
        detection_results["message"] = "Simulation mode (Camera not available on cloud)"

        while is_running:
            # Create simulated frame
            frame = create_placeholder_frame("Camera simulation - Running on cloud")

            # Simulate occasional detections
            if int(time.time()) % 10 == 0:
                detection_results["potholes"] = 1
                location = get_current_location()
                add_pothole_detection(location, 0.8)
                detection_results["message"] = "Simulated pothole detected"
            else:
                detection_results["potholes"] = 0
                detection_results["message"] = "Monitoring road surface (simulation)"

            latest_frame = frame
            time.sleep(1)
        return

    # Original camera code for local deployment
    last_detection_time = 0
    detection_cooldown = 10

    try:
        camera_index = 0
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera = cv2.VideoCapture(camera_index)

        if not camera.isOpened():
            logger.error(f"Could not open camera at index {camera_index}")
            detection_results["status"] = "error"
            detection_results["message"] = "Camera not available"
            latest_frame = create_placeholder_frame("Camera not available")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)

        logger.info("Camera started successfully")
        detection_results["status"] = "running"
        detection_results["message"] = "Detection active"

        while is_running:
            success, frame = camera.read()

            if not success:
                logger.warning("Failed to read frame from camera")
                latest_frame = create_placeholder_frame("Camera read error")
                time.sleep(0.1)
                continue

            display_frame = frame.copy()
            pothole_count = 0
            max_confidence = 0.0

            if model is not None:
                try:
                    results = model(frame, imgsz=640, conf=0.3, verbose=False)

                    for result in results:
                        annotated_frame = result.plot()
                        display_frame = annotated_frame
                        pothole_count = len(result.boxes)

                        if result.boxes and len(result.boxes) > 0:
                            max_confidence = float(result.boxes.conf.max())

                    detection_results["potholes"] = pothole_count

                    if pothole_count > 0:
                        current_time = time.time()
                        if current_time - last_detection_time > detection_cooldown:
                            location = get_current_location()
                            detection_results["current_location"] = location

                            if location:
                                added = add_pothole_detection(location, max_confidence)
                                if added:
                                    last_detection_time = current_time
                                    detection_results["message"] = f"Pothole detected! {pothole_count} found"
                            else:
                                detection_results["message"] = "Pothole detected but location unavailable"
                        else:
                            detection_results["message"] = f"Potholes detected: {pothole_count}"
                    else:
                        detection_results["message"] = "Monitoring road surface"
                        detection_results["current_location"] = get_current_location()

                except Exception as e:
                    logger.error(f"Error in detection: {e}")
                    detection_results["message"] = f"Detection error: {str(e)}"

            latest_frame = display_frame
            time.sleep(0.05)

    except Exception as e:
        logger.error(f"Error in run_detection: {e}")
        detection_results["status"] = "error"
        detection_results["message"] = f"Camera error: {str(e)}"
        latest_frame = create_placeholder_frame(f"Error: {str(e)}")

    finally:
        if camera and camera.isOpened():
            camera.release()
            logger.info("Camera released")
        detection_results["status"] = "stopped"
        detection_results["message"] = "Detection stopped"


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection_data')
def detection_data():
    return jsonify(detection_results)


@app.route('/potholes_list')
def potholes_list():
    return jsonify({
        "potholes": detected_potholes[:20],
        "total_count": len(detected_potholes)
    })


@app.route('/current_location')
def current_location():
    location = get_current_location()
    return jsonify({"location": location})


@app.route('/start_detection')
def start_detection():
    global is_running

    if not is_running:
        is_running = True
        detection_thread = threading.Thread(target=run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        return jsonify({
            "status": "started",
            "message": "Detection started successfully"
        })

    return jsonify({
        "status": "already_running",
        "message": "Detection is already running"
    })


@app.route('/stop_detection')
def stop_detection():
    global is_running, camera, latest_frame

    is_running = False
    if camera and camera.isOpened():
        camera.release()

    latest_frame = create_placeholder_frame("Detection stopped")
    detection_results["potholes"] = 0
    detection_results["status"] = "stopped"
    detection_results["message"] = "Detection stopped"

    return jsonify({
        "status": "stopped",
        "message": "Detection stopped successfully"
    })


@app.route('/clear_potholes')
def clear_potholes():
    global detected_potholes
    detected_potholes = []
    save_pothole_data()
    return jsonify({"message": "Pothole records cleared", "count": 0})


@app.route('/export_potholes')
def export_potholes():
    return jsonify(detected_potholes)


@app.route('/test_camera')
def test_camera():
    if is_render():
        return jsonify({
            "status": "simulated",
            "message": "Camera simulation on cloud - real camera only available locally"
        })

    test_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if test_cam.isOpened():
        success, frame = test_cam.read()
        test_cam.release()
        if success:
            return jsonify({"status": "working", "message": "Camera is working"})

    return jsonify({"status": "error", "message": "Camera not available"})


if __name__ == '__main__':
    load_pothole_data()
    latest_frame = create_placeholder_frame("Click Start Detection")

    logger.info("Starting Pothole Detection System with Geolocation")
    logger.info(f"Loaded {len(detected_potholes)} previous detections")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)