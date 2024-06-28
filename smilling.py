from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import dlib
import numpy as np
import face_recognition
import os
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load pre-trained models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load known faces
def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []

    for image_path in os.listdir(directory):
        if image_path.endswith(".jpg") or image_path.endswith(".png"):
            image = face_recognition.load_image_file(f"{directory}/{image_path}")
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(image_path)[0])

    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces("test")

def estimate_head_pose(landmarks, image_width, image_height):
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),       # Chin
        (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
    ], dtype="double")

    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [angle[0] for angle in euler_angles]

    if yaw < -15:
        return "Left"
    elif yaw > 15:
        return "Right"
    else:
        return "Center"

movements = ["Right", "Left"]
current_movement = 0

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_movement

    file = request.files['frame']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    frame = np.array(img)

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_person = "Unmatch"
    head_orientation = "Unknown"

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unmatch"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            current_person = name

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_detector(gray)

        for face_rect in face_rects:
            landmarks = landmark_predictor(gray, face_rect)
            head_orientation = estimate_head_pose(landmarks, frame.shape[1], frame.shape[0])

            if head_orientation == movements[current_movement]:
                current_movement += 1
                if current_movement >= len(movements):
                    current_movement = 0  # Reset for next cycle

    response = {
        "recognized_person": current_person,
        "current_movement": movements[current_movement] if current_movement < len(movements) else "Completed",
        "head_orientation": head_orientation
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
