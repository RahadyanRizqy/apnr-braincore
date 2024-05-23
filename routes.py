from flask import Blueprint, render_template, request, flash, send_from_directory
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
import cv2
from ultralytics import YOLO

from collections import Counter

load_dotenv()

bp = Blueprint("routes", __name__)

@bp.route("/")
def index():
    return render_template('home.html', title='Smart APNR');


# @bp.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(bp.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'mov', 'avi', 'wmv', 'flv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/process', methods=['POST'])
def process():
    try:
        if request.method == 'POST':
            file = request.file['video_file']
            if file and allowed_file(file.filename):
                _filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, _filename)
                file.save(file_path)
            # NEED MODEL
    except Exception as e:
        return f"<h1>Error {str(e)} </h1>"


# @bp.route("/analyze", methods=["POST"])
# def analyze():
#     try:
#         if request.method == "POST":
#             file = request.files["audio_file"]
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(file_path)


#             else:
#                 flash("Please upload proper wav/mp3 file", "warning")
#                 return render_template("pages/index.html")
#     except Exception as e:
#         flash(str(e))
#         return render_template("pages/errors/500.html")



# Load YOLO model for face detection
model_yolo = YOLO("models/license_plate_detector.pt")

# Function to detect faces using YOLO from webcam
def detect_faces_with_yolo():
    cap = cv2.VideoCapture(0)  # Using the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using YOLO
        results = model_yolo(frame)
        results = results[0]

        # Draw bounding boxes and labels for detected faces
        for box in results.boxes:
            confidence = box.conf[0].item()
            if confidence > 0.6:
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_id = box.cls[0].item()
                label = results.names[class_id]
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the modified frame
        cv2.imshow('Webcam', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the face detection function
detect_faces_with_yolo()