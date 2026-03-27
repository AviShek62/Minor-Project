import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
try:
    from playsound import playsound
    from threading import Thread
except ImportError:
    print("Warning: playsound not available, using system beep")
    playsound = None

try:
    import winsound
except ImportError:
    print("Warning: winsound not available")
    winsound = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def start_alarm(sound):
    if playsound:
        try:
            playsound(sound)
        except:
            print("AUDIO ALERT: DROWSINESS DETECTED!")
    else:
        print("AUDIO ALERT: DROWSINESS DETECTED!")

def initialize_camera():
    """Initialize camera with multiple backends"""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        print(f"Trying camera backend: {backend}")
        cap = cv2.VideoCapture(0, backend)
        
        if cap.isOpened():
            # Test reading a frame
            for i in range(5):  # Try multiple times
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Camera initialized successfully with backend {backend}")
                    return cap
                time.sleep(0.1)
        
        cap.release()
    
    print("Failed to initialize camera with all backends")
    return None

# Fixed class mapping based on actual behavior:
# Class 0: Closed eyes
# Class 1: Open eyes 
# Class 2: Closed/Drowsy eyes (commonly predicted when eyes closed)
# Class 3: Open eyes (commonly predicted when eyes open)
classes = ['Closed', 'Open', 'Closed', 'Open']

print("Loading cascade classifiers...")
face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "data", "haarcascade_frontalface_default.xml")
)
left_eye_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "data", "haarcascade_lefteye_2splits.xml")
)
right_eye_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "data", "haarcascade_righteye_2splits.xml")
)

if face_cascade.empty() or left_eye_cascade.empty() or right_eye_cascade.empty():
    print("Error: Could not load cascade files")
    exit()

print("Loading model...")
model = load_model(os.path.join(BASE_DIR, "drowiness_new7.keras"))
print("Model loaded successfully!")

print("Initializing camera...")
cap = initialize_camera()

if cap is None:
    print("\nCamera initialization failed!")
    print("Possible solutions:")
    print("1. Check if camera is connected")
    print("2. Close other applications using camera")
    print("3. Grant camera permissions in Windows Settings")
    print("4. Update camera drivers")
    print("\nRunning in demo mode instead...")
    
    # Demo mode with simulated detection
    demo_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    cv2.circle(demo_frame, (320, 240), 80, (200, 200, 200), -1)
    cv2.circle(demo_frame, (295, 220), 15, (50, 50, 50), -1)
    cv2.circle(demo_frame, (345, 220), 15, (50, 50, 50), -1)
    
    cv2.putText(demo_frame, "DEMO MODE - No Camera", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(demo_frame, "Press 'q' to quit", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    while True:
        cv2.imshow("Drowsiness Detector - Demo Mode", demo_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    exit()

print("Camera ready! Starting detection...")
print("Press 'q' to quit")

count = 0
alarm_on = False
alarm_sound = os.path.join(BASE_DIR, "data", "alarm.mp3")
status1 = 1
status2 = 1
frame_count = 0
failed_reads = 0

while True:
    ret, frame = cap.read()
    if not ret:
        failed_reads += 1
        if failed_reads > 10:
            print("Too many failed frame reads, exiting...")
            break
        time.sleep(0.1)
        continue
    
    failed_reads = 0
    frame_count += 1

    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Initialize eye status (1 = open, 0 = closed)
    status1, status2 = 1, 1
    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes with more sensitive parameters
        left_eye = left_eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))
        right_eye = right_eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))

        # Process left eye
        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            if eye1.shape[0] > 10 and eye1.shape[1] > 10:  # Valid eye region
                eye1 = cv2.resize(eye1, (145, 145))
                eye1 = eye1.astype('float') / 255.0
                eye1 = img_to_array(eye1)
                eye1 = np.expand_dims(eye1, axis=0)
                pred1 = model.predict(eye1, verbose=0)
                status1 = np.argmax(pred1)
                eyes_detected = True
                print(f"Left eye prediction: {pred1[0]} -> {status1} ({classes[status1] if status1 < len(classes) else 'Unknown'})")
                # Show prediction confidence
                if status1 < len(classes):
                    cv2.putText(frame, f"L: {classes[status1]} ({np.max(pred1):.2f})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            break

        # Process right eye
        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2+h2, x2:x2+w2]
            if eye2.shape[0] > 10 and eye2.shape[1] > 10:  # Valid eye region
                eye2 = cv2.resize(eye2, (145, 145))
                eye2 = eye2.astype('float') / 255.0
                eye2 = img_to_array(eye2)
                eye2 = np.expand_dims(eye2, axis=0)
                pred2 = model.predict(eye2, verbose=0)
                status2 = np.argmax(pred2)
                eyes_detected = True
                print(f"Right eye prediction: {pred2[0]} -> {status2} ({classes[status2] if status2 < len(classes) else 'Unknown'})")
                # Show prediction confidence
                if status2 < len(classes):
                    cv2.putText(frame, f"R: {classes[status2]} ({np.max(pred2):.2f})", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            break

    # Drowsiness detection logic
    if eyes_detected:
        print(f"Eye status: L={status1}, R={status2}, Count={count}")
        # Fix inverted mapping: class 2 = closed/drowsy, class 3 = open
        closed_left = (status1 == 0 or status1 == 2)
        closed_right = (status2 == 0 or status2 == 2)
        
        if closed_left and closed_right:  # Both eyes closed/drowsy
            count += 1
            print(f"BOTH EYES CLOSED! Count: {count}")
            cv2.putText(frame, "Eyes Closed", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            if count >= 5:  # Reduced threshold for faster detection
                cv2.putText(frame, "DROWSINESS ALERT!!!",
                            (50, height - 100),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)

                if not alarm_on:
                    alarm_on = True
                    print("*** DROWSINESS DETECTED! ALARM RINGING! ***")
                
                # Keep playing alarm continuously while eyes are closed
                try:
                    if playsound and os.path.exists(alarm_sound):
                        # Play alarm in separate thread so it doesn't block detection
                        t = Thread(target=start_alarm, args=(alarm_sound,))
                        t.daemon = True
                        t.start()
                    else:
                        # Fallback: system beep
                        if winsound:
                            winsound.Beep(1000, 300)  # 1000Hz for 300ms
                        else:
                            print("\a*** WAKE UP! DROWSINESS DETECTED! ***")
                except Exception as e:
                    print("*** WAKE UP! DROWSINESS DETECTED! ***")
        elif closed_left or closed_right:  # One eye closed
            cv2.putText(frame, "One Eye Closed", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255), 2)
            count = max(0, count - 1)  # Slowly decrease count
        else:  # Both eyes open
            if alarm_on:
                print("*** EYES OPENED - ALARM STOPPED ***")
            count = 0
            alarm_on = False
            cv2.putText(frame, "Eyes Open", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Eyes Detected", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        count = max(0, count - 1)  # Slowly decrease count when no eyes detected

    # Add status information
    cv2.putText(frame, f"Faces: {len(faces)} | Eyes: {'Yes' if eyes_detected else 'No'}", 
                (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Closed Count: {count}/5", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    left_status = classes[status1] if eyes_detected and status1 < len(classes) else 'N/A'
    right_status = classes[status2] if eyes_detected and status2 < len(classes) else 'N/A'
    cv2.putText(frame, f"Status: L={left_status} R={right_status}", 
                (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
