# import cv2
# import threading
# import numpy as np
# import pyaudio
# import librosa
# import joblib
# import time
# import pyttsx3
# from ultralytics import YOLO

# # here we are loading the model (best one from runs)
# model_path = "C:\\Users\\Koushik N\\Desktop\\Yolo11_Final_Project\\runs\\detect\\train\\weights\\best.pt"
# yolo_model = YOLO(model_path)

# # here we are loading models for siren detection
# siren_model = joblib.load("siren_detection_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # initialising text-to-speech engine
# engine = pyttsx3.init()
# engine.setProperty("rate", 150)

# # Audio config
# CHUNK = 2048
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100

# # Siren detection control
# siren_running = False
# siren_thread = None

# # Feature extraction
# def extract_features(audio_data, sr=RATE):
#     try:
#         mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
#         chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
#         contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
#         return np.hstack([
#             np.mean(mfccs, axis=1),
#             np.mean(chroma, axis=1),
#             np.mean(contrast, axis=1)
#         ])
#     except:
#         return None

# # Siren detection logic
# def siren_detection_worker():
#     global siren_running
#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
#                     input=True, frames_per_buffer=CHUNK)
#     siren_detected = False
#     silent_counter = 0

#     print("🔊 Siren detection started.")
#     while siren_running:
#         try:
#             data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
#             data = data.astype(float) / 32768.0
#             features = extract_features(data)
#             if features is None:
#                 continue
#             features = scaler.transform(features.reshape(1, -1))
#             prob = siren_model.predict_proba(features)[0][1]

#             if prob > 0.6:
#                 if not siren_detected:
#                     print("🚨 Siren Detected!")
#                     engine.say("Emergency vehicle detected. Please slow down and give way.")
#                     engine.runAndWait()
#                     siren_detected = True
#                     silent_counter = 0
#             else:
#                 silent_counter += 1
#                 if silent_counter > 2:
#                     siren_detected = False
#                     silent_counter = 0

#             time.sleep(0.2)
#         except Exception as e:
#             print(f"Error in siren detection: {e}")
#             break

#     print("🛑 Siren detection stopped.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

# # Start siren thread
# def start_siren_detection():
#     global siren_running, siren_thread
#     if not siren_running:
#         siren_running = True
#         siren_thread = threading.Thread(target=siren_detection_worker)
#         siren_thread.start()

# # Stop siren thread
# def stop_siren_detection():
#     global siren_running, siren_thread
#     if siren_running:
#         siren_running = False
#         if siren_thread is not None:
#             siren_thread.join()

# # Open webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# print("🚀 Press 'd' to START siren detection | 'f' to STOP | 'q' to quit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Run YOLOv11 model
#     results = yolo_model(frame)

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls = int(box.cls[0].item())
#             class_name = yolo_model.names[cls]

#             # Draw and speak
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, class_name, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             engine.say(class_name)
#             engine.runAndWait()

#     cv2.imshow("YOLOv11 Real-Time Detection", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('d'):
#         start_siren_detection()
#     elif key == ord('f'):
#         stop_siren_detection()

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# stop_siren_detection()


#this is final one 25/04/25 

# import cv2
# import threading
# import numpy as np
# import pyaudio
# import librosa
# import joblib
# import time
# import pyttsx3
# from ultralytics import YOLO

# # Load YOLO model
# model_path = "C:\\Users\\Koushik N\\Desktop\\Yolo11_Final_Project\\runs\\detect\\train\\weights\\best.pt"
# yolo_model = YOLO(model_path)

# # Load siren detection model and scaler
# siren_model = joblib.load("siren_detection_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # Initialize TTS engine
# engine = pyttsx3.init()
# engine.setProperty("rate", 150)
# speech_lock = threading.Lock()

# # Function to speak text without blocking main thread
# def speak(text):
#     def run():
#         with speech_lock:
#             engine.say(text)
#             engine.runAndWait()
#     threading.Thread(target=run).start()

# # Audio configuration
# CHUNK = 2048
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100

# # Siren detection control
# siren_running = False
# siren_thread = None

# # Extract features from audio
# def extract_features(audio_data, sr=RATE):
#     try:
#         mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
#         chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
#         contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
#         return np.hstack([
#             np.mean(mfccs, axis=1),
#             np.mean(chroma, axis=1),
#             np.mean(contrast, axis=1)
#         ])
#     except:
#         return None

# # Siren detection thread logic
# def siren_detection_worker():
#     global siren_running
#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
#                     input=True, frames_per_buffer=CHUNK)
#     siren_detected = False
#     silent_counter = 0

#     print("🔊 Siren detection started.")
#     while siren_running:
#         try:
#             data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
#             data = data.astype(float) / 32768.0
#             features = extract_features(data)
#             if features is None:
#                 continue
#             features = scaler.transform(features.reshape(1, -1))
#             prob = siren_model.predict_proba(features)[0][1]

#             if prob > 0.6:
#                 if not siren_detected:
#                     print("🚨 Siren Detected!")
#                     speak("Emergency vehicle detected. Please slow down and give way.")
#                     siren_detected = True
#                     silent_counter = 0
#             else:
#                 silent_counter += 1
#                 if silent_counter > 2:
#                     siren_detected = False
#                     silent_counter = 0

#             time.sleep(0.2)
#         except Exception as e:
#             print(f"Error in siren detection: {e}")
#             break

#     print("🛑 Siren detection stopped.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

# # Start siren thread
# def start_siren_detection():
#     global siren_running, siren_thread
#     if not siren_running:
#         siren_running = True
#         siren_thread = threading.Thread(target=siren_detection_worker)
#         siren_thread.start()

# # Stop siren thread
# def stop_siren_detection():
#     global siren_running, siren_thread
#     if siren_running:
#         siren_running = False
#         if siren_thread is not None:
#             siren_thread.join()

# # Open webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# print("🚀 Press 'd' to START siren detection | 'f' to STOP | 'q' to quit")

# last_spoken = {}
# speak_interval = 1  # seconds

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Run YOLO model
#     results = yolo_model(frame)

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls = int(box.cls[0].item())
#             class_name = yolo_model.names[cls]

#             # Draw bounding box and label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, class_name, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Speak class name once per second
#             current_time = time.time()
#             if class_name not in last_spoken or (current_time - last_spoken[class_name]) > speak_interval:
#                 speak(class_name)
#                 last_spoken[class_name] = current_time

#     cv2.imshow("YOLOv11 Real-Time Detection", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('d'):
#         start_siren_detection()
#     elif key == ord('f'):
#         stop_siren_detection()

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# stop_siren_detection()



import cv2
from ultralytics import YOLO
import pyttsx3  # Text-to-speech library

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the YOLOv11 model
model_path = "C:\\Users\\Koushik N\\Desktop\\Yolo11_Final_Project\\runs\\detect\\train\\weights\\best.pt"
model = YOLO(model_path)

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Run YOLOv11 model on the frame
    results = model(frame)
    
    # Display the results on the frame
    for result in results:
         for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            class_name = model.names[cls]
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Read aloud only the class name
            engine.say(class_name)  # Speaking only the class name
            engine.runAndWait()  # Wait until the speech is completed
    
    # Show the frame
    cv2.imshow("YOLOv11 Real-Time Detection", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release resources
cap.release()
cv2.destroyAllWindows()