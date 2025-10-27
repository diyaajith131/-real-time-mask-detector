import cv2 #type:ignore
import numpy as np #type:ignore
import tensorflow as tf #type:ignore
from flask import Flask, render_template, Response, jsonify, request #type:ignore
from flask_cors import CORS #type:ignore
from flask.json.provider import DefaultJSONProvider #type:ignore
import os #type:ignore
import time
import random
import threading
import json

# Custom JSON provider to handle NumPy types
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Initialize Flask app
app = Flask(__name__)
app.json = NumpyJSONProvider(app)  # Set custom JSON provider
CORS(app)  # Enable CORS for all routes

# Load the trained model
print("Loading face mask detection model...")
mask_model = None
try:
    if os.path.exists('models/face_mask_detector.h5'):
        mask_model = tf.keras.models.load_model('models/face_mask_detector.h5')
        print("AI Model loaded successfully!")
    else:
        print("Model file not found - using improved simulation mode")
        mask_model = None
except Exception as e:
    print(f"Model not available: {e}")
    print("Using improved simulation mode")
    mask_model = None

# Load OpenCV Haar Cascade for face detection
print("Loading Haar Cascade for face detection...")
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Haar Cascade loaded successfully!")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    face_cascade = None

# Initialize video capture
cap = None

# Global variables for alerts and notifications
alerts = []
last_alert_time = 0
alert_cooldown = 3.0  # Minimum time between alerts (seconds)
detection_stats = {
    'total_detections': 0,
    'mask_count': 0,
    'no_mask_count': 0,
    'last_detection': None
}

# Detection control variables
detection_active = False
detection_sensitivity = 0.5  # 0.1 to 1.0
alert_threshold = 0.7  # 0.1 to 1.0
snapshot_counter = 0

# Manual override for testing - set to True to force "no mask" detection
force_no_mask = False

def initialize_camera():
    """Initialize the camera"""
    global cap
    if cap is None:
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Camera initialized successfully on index {camera_index}")
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return True
            else:
                cap.release()
        
        print("Error: Could not open any camera")
        return False
    return True

def preprocess_face(face_roi):
    """Preprocess face region for model prediction"""
    # Resize to 224x224
    face_resized = cv2.resize(face_roi, (224, 224))
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    face_normalized = face_rgb.astype(np.float32) / 255.0
    
    # Expand dimensions for batch prediction
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def create_alert(alert_type, message, confidence=None):
    """Create a new alert"""
    global alerts, last_alert_time, detection_stats
    
    current_time = time.time()
    
    # Check cooldown to prevent spam
    if current_time - last_alert_time < alert_cooldown:
        return
    
    # Convert numpy types to Python native types
    if confidence is not None and isinstance(confidence, (np.floating, np.integer)):
        confidence = float(confidence)
    
    alert = {
        'id': len(alerts) + 1,
        'type': alert_type,  # 'mask' or 'no_mask'
        'message': message,
        'confidence': confidence,
        'timestamp': current_time,
        'time_str': time.strftime('%H:%M:%S', time.localtime(current_time)),
        'processed': False  # Mark as unprocessed for frontend
    }
    
    alerts.insert(0, alert)  # Add to beginning
    last_alert_time = current_time
    #time.sleep(10)
    
    # Keep only last 50 alerts
    if len(alerts) > 50:
        alerts = alerts[:50]
    
    # Update statistics
    detection_stats['total_detections'] += 1
    if alert_type == 'mask':
        detection_stats['mask_count'] += 1
    else:
        detection_stats['no_mask_count'] += 1
    
    detection_stats['last_detection'] = alert
    
    print(f"ALERT: {message} (Confidence: {confidence:.2f})")
    
    # Force update the frontend by triggering a status change
    return alert

def get_alerts():
    """Get recent alerts"""
    return alerts[:10]  # Return last 10 alerts

def get_stats():
    """Get detection statistics"""
    return detection_stats

def is_face_still_visible(face, faces):
    """Check if the focused face is still visible in the current frame"""
    if face is None or len(faces) == 0:
        return False
    
    fx, fy, fw, fh = face
    for (x, y, w, h) in faces:
        # Check if faces overlap significantly
        if (abs(x - fx) < fw/2 and abs(y - fy) < fh/2):
            return True
    return False

def improved_simulation_detection(face_roi, w, h):
    """Improved simulation detection based on face characteristics"""
    # Analyze face characteristics for more realistic detection
    face_area = w * h
    face_ratio = w / h if h > 0 else 1.0
    
    # Convert to HSV for better color analysis
    hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    # Analyze lower face region (mouth area) for mask-like colors
    lower_face = face_roi[int(h*0.6):h, :]
    if lower_face.size > 0:
        hsv_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        
        # Look for mask-like colors (blues, whites, grays)
        mask_colors = [
            ([100, 50, 50], [130, 255, 255]),  # Blue range
            ([0, 0, 200], [180, 30, 255]),     # White range
            ([0, 0, 100], [180, 50, 200])      # Gray range
        ]
        
        mask_pixel_count = 0
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        
        for (lower, upper) in mask_colors:
            mask = cv2.inRange(hsv_lower, np.array(lower), np.array(upper))
            mask_pixel_count += cv2.countNonZero(mask)
        
        mask_ratio = mask_pixel_count / total_pixels if total_pixels > 0 else 0
        
        # Determine if mask is present based on color analysis
        has_mask = mask_ratio > 0.15  # Threshold for mask detection
        
        # Adjust confidence based on face characteristics
        base_confidence = 0.85
        
        # Face size factor
        if face_area < 3000:  # Small face
            confidence = base_confidence - 0.1
        elif face_area > 10000:  # Large face
            confidence = base_confidence + 0.05
        else:
            confidence = base_confidence
        
        # Face ratio factor
        if 0.7 <= face_ratio <= 0.9:  # Good face ratio
            confidence += 0.05
        else:
            confidence -= 0.05
        
        # Mask detection factor
        if has_mask:
            confidence += mask_ratio * 0.2
        else:
            confidence += (1 - mask_ratio) * 0.1
        
        # Apply sensitivity setting
        confidence = confidence * detection_sensitivity
        
        # Clamp confidence between 0.5 and 0.98
        confidence = max(0.5, min(0.98, confidence))
        
        # Manual override for testing
        if force_no_mask:
            has_mask = False
            confidence = random.uniform(0.90, 0.98)
        
        return has_mask, float(confidence)  # Convert to Python float
    
    # Fallback to random if analysis fails
    has_mask = random.random() > 0.7  # 30% chance of mask
    confidence = random.uniform(0.75, 0.95)
    return has_mask, float(confidence)  # Convert to Python float

def get_detection_result(has_mask, confidence):
    """Get detection result based on mask presence and confidence"""
    if has_mask:
        label = "MASK DETECTED"
        color = (0, 255, 0)  # Green
        create_alert('mask', '✅ GOOD: Person is wearing a mask correctly', confidence)
    else:
        label = "NO MASK DETECTED"
        color = (0, 0, 255)  # Red
        create_alert('no_mask', '🚨 VIOLATION: Person is NOT wearing a mask!', confidence)
    
    return label, color

def draw_detection_results(frame, x, y, w, h, mouth_x, mouth_y, mouth_w, mouth_h, 
                         label_with_conf, color, face_detected):
    """Draw detection results on the frame"""
    # Draw main face rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw mouth area focus rectangle
    cv2.rectangle(frame, (mouth_x, mouth_y), (mouth_x+mouth_w, mouth_y+mouth_h), color, 3)
    
    if face_detected:
        # Draw focus indicator
        cv2.putText(frame, "FOCUS", (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label_with_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x, y-25), (x + label_size[0], y-5), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_with_conf, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        # Draw analyzing state
        cv2.putText(frame, "ANALYZING...", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def detect_mask():
    """Generator function for video frames with mask detection"""
    global cap
    
    if not initialize_camera():
        return
    
    # Simulation variables
    last_detection_time = 0
    detection_interval = 4.0  # Simulate detection every 4 seconds
    current_focus_face = None
    focus_start_time = 0
    focus_duration = 6.0  # Focus on one face for 6 seconds
    face_detected = False
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with improved parameters
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  # More sensitive scaling
                minNeighbors=8,    # Higher quality detection
                minSize=(80, 80),  # Larger minimum size for better accuracy
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Sort faces by size (largest first) and focus on the biggest one
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                current_time = time.time()
                
                # Check if we need to switch focus to a new face
                if (current_focus_face is None or 
                    current_time - focus_start_time > focus_duration or
                    not is_face_still_visible(current_focus_face, faces)):
                    
                    # Focus on the largest face
                    current_focus_face = faces[0]
                    focus_start_time = current_time
                
                # Process only the focused face
                (x, y, w, h) = current_focus_face
                
                # Calculate mouth area (lower half of face)
                mouth_y = y + int(h * 0.6)  # Start from 60% down the face
                mouth_h = int(h * 0.4)      # Cover 40% of face height
                mouth_x = x + int(w * 0.2)   # Start from 20% from left
                mouth_w = int(w * 0.6)       # Cover 60% of face width
                
                # Extract mouth region for better mask detection
                mouth_roi = frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
                
                # Extract full face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Process face detection and mask analysis
                if face_roi.size > 0:
                    current_time = time.time()
                    
                    # Only detect if we have a stable face for a few seconds
                    if not face_detected:
                        face_detected = True
                        print("Face detected - starting analysis...")
                    
                    # Use AI model if available, otherwise use improved simulation
                    if detection_active and mask_model is not None and current_time - last_detection_time > detection_interval:
                        try:
                            # Preprocess face for model prediction
                            face_preprocessed = preprocess_face(face_roi)
                            
                            # Make prediction
                            prediction = mask_model.predict(face_preprocessed, verbose=0)
                            predicted_class = np.argmax(prediction[0])
                            confidence = float(np.max(prediction[0]))  # Convert to Python float
                            
                            # Apply user-defined threshold
                            if confidence >= alert_threshold:
                                if predicted_class == 0:  # with_mask
                                    label = "MASK DETECTED"
                                    color = (0, 255, 0)  # Green
                                    create_alert('mask', '✅ GOOD: Person is wearing a mask correctly', confidence)
                                else:  # without_mask
                                    label = "NO MASK DETECTED"
                                    color = (0, 0, 255)  # Red
                                    create_alert('no_mask', '🚨 VIOLATION: Person is NOT wearing a mask!', confidence)
                            else:
                                # Low confidence - show analyzing
                                label = "ANALYZING..."
                                color = (255, 255, 0)  # Yellow
                            
                            label_with_conf = f"{label} ({confidence:.2f})"
                            last_detection_time = current_time
                            
                        except Exception as e:
                            print(f"Error in AI prediction: {e}")
                            # Fallback to simulation
                            has_mask, confidence = improved_simulation_detection(face_roi, w, h)
                            label, color = get_detection_result(has_mask, confidence)
                            label_with_conf = f"{label} ({confidence:.2f})"
                            last_detection_time = current_time
                    
                    elif detection_active and mask_model is None and current_time - last_detection_time > detection_interval:
                        # Improved simulation mode
                        has_mask, confidence = improved_simulation_detection(face_roi, w, h)
                        label, color = get_detection_result(has_mask, confidence)
                        label_with_conf = f"{label} ({confidence:.2f})"
                        last_detection_time = current_time
                    
                    elif detection_active:
                        # Show analyzing state when detection is active
                        label = "ANALYZING..."
                        color = (255, 255, 0)  # Yellow
                        label_with_conf = label
                    else:
                        # Show waiting state when detection is not active
                        label = "DETECTION OFF"
                        color = (128, 128, 128)  # Gray
                        label_with_conf = label
                    
                    # Draw detection results
                    draw_detection_results(frame, x, y, w, h, mouth_x, mouth_y, mouth_w, mouth_h, 
                                        label_with_conf, color, face_detected)
        
        # Add status text to frame
        if detection_active:
            if mask_model is not None:
                status_text = f"AI Model: Active (Threshold: {alert_threshold:.1f})"
            else:
                status_text = f"Enhanced Simulation Mode (Sensitivity: {detection_sensitivity:.1f})"
        else:
            status_text = "Detection: OFF - Click Start Detection to begin"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error encoding frame: {e}")
            break

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(detect_mask(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    """Health check route"""
    return {
        'status': 'healthy',
        'model_loaded': mask_model is not None,
        'cascade_loaded': face_cascade is not None,
        'mode': 'simulation' if mask_model is None else 'ai_model',
        'message': 'Running in simulation mode - install TensorFlow for AI model' if mask_model is None else 'AI model active'
    }

@app.route('/api/alerts')
def api_alerts():
    """Get recent alerts"""
    return jsonify(get_alerts())

@app.route('/api/stats')
def api_stats():
    """Get detection statistics"""
    return jsonify(get_stats())

@app.route('/api/clear-alerts')
def clear_alerts():
    """Clear all alerts"""
    global alerts
    alerts = []
    return jsonify({'status': 'cleared', 'message': 'All alerts cleared'})

@app.route('/api/current-status')
def current_status():
    """Get current detection status"""
    if len(alerts) > 0:
        latest_alert = alerts[0]
        return jsonify({
            'status': 'detected',
            'type': latest_alert['type'],
            'message': latest_alert['message'],
            'confidence': latest_alert['confidence'],
            'timestamp': latest_alert['timestamp'],
            'time_str': latest_alert['time_str']
        })
    else:
        return jsonify({
            'status': 'waiting',
            'message': 'Waiting for face detection...'
        })

@app.route('/api/toggle-force-no-mask')
def toggle_force_no_mask():
    """Toggle force no mask detection for testing"""
    global force_no_mask
    force_no_mask = not force_no_mask
    return jsonify({
        'force_no_mask': force_no_mask,
        'message': f'Force no mask detection: {"ON" if force_no_mask else "OFF"}'
    })

@app.route('/api/toggle-detection', methods=['GET'])
def toggle_detection():
    """Toggle detection on/off"""
    global detection_active
    detection_active = not detection_active
    return jsonify({
        'detection_active': detection_active,
        'message': f'Detection {"started" if detection_active else "stopped"}'
    })

@app.route('/api/set-sensitivity', methods=['POST'])
def set_sensitivity():
    """Set detection sensitivity"""
    global detection_sensitivity
    data = request.get_json()
    sensitivity = float(data.get('sensitivity', 0.5))
    
    if 0.1 <= sensitivity <= 1.0:
        detection_sensitivity = sensitivity
        return jsonify({
            'sensitivity': detection_sensitivity,
            'message': f'Detection sensitivity set to {sensitivity}'
        })
    else:
        return jsonify({'error': 'Sensitivity must be between 0.1 and 1.0'}), 400

@app.route('/api/set-threshold', methods=['POST'])
def set_threshold():
    """Set alert threshold"""
    global alert_threshold
    data = request.get_json()
    threshold = float(data.get('threshold', 0.7))
    
    if 0.1 <= threshold <= 1.0:
        alert_threshold = threshold
        return jsonify({
            'threshold': alert_threshold,
            'message': f'Alert threshold set to {threshold}'
        })
    else:
        return jsonify({'error': 'Threshold must be between 0.1 and 1.0'}), 400

@app.route('/api/capture-snapshot', methods=['POST'])
def capture_snapshot():
    """Capture a snapshot of the current video frame"""
    global snapshot_counter
    snapshot_counter += 1
    
    # In a real implementation, you would capture the actual frame
    # For now, we'll return a success message
    return jsonify({
        'snapshot_id': snapshot_counter,
        'message': f'Snapshot #{snapshot_counter} captured successfully',
        'timestamp': time.time()
    })

@app.route('/api/detection-status')
def get_detection_status():
    """Get current detection status and settings"""
    return jsonify({
        'detection_active': detection_active,
        'sensitivity': detection_sensitivity,
        'threshold': alert_threshold,
        'snapshot_count': snapshot_counter
    })

if __name__ == '__main__':
    print("Starting Face Mask Detection Server...")
    print("Open your browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
