from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import dlib
from imutils import face_utils

app = Flask(__name__)
CORS(app)

# Load the image to recognize
known_image_path = "amine.jpg"  # replace with your image path
known_image = face_recognition.load_image_file(known_image_path)
known_face_encodings = face_recognition.face_encodings(known_image)

if len(known_face_encodings) > 0:
    known_face_encoding = known_face_encodings[0]
else:
    raise ValueError("No faces found in the known image.")

# Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # replace with the path to the predictor

def get_head_orientation(shape, frame):
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left Mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-165.0, 170.0, -135.0),     # Left eye left corner
        (165.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    size = (frame.shape[1], frame.shape[0])
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
    angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[6]
    return angles.flatten()

def detect_liveness(frame, required_movement):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        angles = get_head_orientation(shape, frame)

        if required_movement == "center":
            return bool(-15 < angles[1] < 15), rect
        elif required_movement == "left":
            return bool(angles[1] > 15), rect
        elif required_movement == "right":
            return bool(angles[1] < -15), rect

    return False, None

@app.route('/video_feed', methods=['POST'])
def video_feed():
    if 'video' not in request.files:
        return jsonify({"error": "No video part"}), 400

    file = request.files['video']
    video = file.read()

    required_movement = request.form['movement']

    nparr = np.frombuffer(video, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    is_real, face_location = detect_liveness(frame, required_movement)

    if face_location is not None:
        face_location_dict = {
            "top": face_location.top(),
            "right": face_location.right(),
            "bottom": face_location.bottom(),
            "left": face_location.left()
        }
    else:
            face_location_dict = None

    response = {
        "is_real": bool(is_real),
        "face_location": face_location_dict
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
