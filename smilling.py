import cv2
import dlib
import numpy as np
import face_recognition
import os

# Load pre-trained models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def estimate_head_pose(landmarks, image_width, image_height):
    # Define 3D model points of the face for pose estimation
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera intrinsic parameters (approximation)
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # Distortion coefficients (assuming no distortion)
    dist_coeffs = np.zeros((4, 1))

    # 2D image points from landmarks
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),       # Chin
        (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
    ], dtype="double")

    # Solve PnP to get rotation and translation vectors
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    # Extract Euler angles
    pitch, yaw, roll = [angle[0] for angle in euler_angles]

    # Determine head orientation based on Euler angles
    if yaw < -15:
        return "Left"
    elif yaw > 15:
        return "Right"
    elif pitch < -15:
        return "Up"
    elif pitch > 15:
        return "Down"
    else:
        return "Center"

def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []

    # Assuming you have images in the format 'name.jpg' in the directory
    for image_path in os.listdir(directory):
        if image_path.endswith(".jpg") or image_path.endswith(".png"):
            image = face_recognition.load_image_file(f"{directory}/{image_path}")
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(image_path)[0])

    return known_face_encodings, known_face_names

def main():
    known_face_encodings, known_face_names = load_known_faces("test")

    cap = cv2.VideoCapture(0)
    movements = ["Right", "Left", "Up", "Down"]
    current_movement = 0
    detected_movements = []
    current_person = None

    while current_movement < len(movements):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unmatch"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                current_person = name  # Track the current recognized person

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = face_detector(gray)

            for face_rect in face_rects:
                landmarks = landmark_predictor(gray, face_rect)
                head_orientation = estimate_head_pose(landmarks, frame.shape[1], frame.shape[0])

                if head_orientation == movements[current_movement]:
                    print(f"Turned {movements[current_movement]}")
                    cv2.putText(frame, f"Turned {movements[current_movement]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    detected_movements.append(movements[current_movement])
                    current_movement += 1
                    break

                for i in range(68):
                    cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0, 0, 255), -1)

            if current_movement >= len(movements):
                break

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After detecting all movements, proceed to match/unmatch verification
    if len(detected_movements) == len(movements) and current_person is not None:
        print(f"All movements detected successfully for {current_person}. Proceeding to match verification.")

        # Continue with face recognition and matching logic here
        # Replace this with your actual logic to determine match/unmatch based on current_person

        # For demonstration purposes, assume it's a match if the person's name is known
        if current_person in known_face_names:
            print("Match")
        else:
            print("Unmatch")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
