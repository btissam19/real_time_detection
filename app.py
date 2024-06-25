import cv2
import face_recognition
import time
import os
import dlib
from scipy.spatial import distance as dist

# Load the reference image and encode the face
reference_image_path = 'ibtissam.jpg'
reference_image = face_recognition.load_image_file(reference_image_path)
reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Directory to save images
if not os.path.exists('captures'):
    os.makedirs('captures')

# Function to capture and save image
def capture_image(frame, label):
    image_path = f"captures/{label}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(image_path, frame)
    print(f"Image saved to {image_path}")
    return image_path

# Constants for blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Initialize variables for blink detection
COUNTER = 0
TOTAL = 0
match_found = False

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Start time to handle timeout for unmatch status
start_time = time.time()
max_time = 60  # Maximum time in seconds to wait for a match

# Main loop to capture video and process instructions
while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_recognition.face_landmarks(rgb_frame)

        if shape:
            # Extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[0]['left_eye']
            rightEye = shape[0]['right_eye']
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # Check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                # If the eyes were closed for a sufficient number of frames, then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # Reset the eye frame counter
                COUNTER = 0

            # Draw the total number of blinks on the frame along with the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

            if face_landmarks_list:
                # Check if face matches reference image
                face_encoding = face_recognition.face_encodings(rgb_frame)
                if len(face_encoding) > 0:
                    match = face_recognition.compare_faces([reference_face_encoding], face_encoding[0])
                    if match[0] and TOTAL >= 1:  # Ensure at least one blink detected
                        match_found = True
                        cv2.putText(frame, "Match Found!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        capture_image(frame, "match")
                        print("Match found with reference image!")
                        break

    cv2.imshow('Video', frame)

    if match_found:
        break

    if time.time() - start_time > max_time:
        capture_image(frame, "unmatch")
        print("Unmatch status: No match found within the time limit.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
