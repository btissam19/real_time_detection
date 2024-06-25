import cv2
import face_recognition
import time
import os

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
def capture_image(frame, step):
    image_path = f"captures/{step}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(image_path, frame)
    print(f"Image saved to {image_path}")
    return image_path

# Instructions and corresponding steps
instructions = [
    "Please look to your left.",
    "Please look to your right.",
    "Please face forward and smile."
]
steps = ["left", "right", "forward"]

# Initialize variables
current_step = 0
processing_step = True
match_found = False

# Main loop to capture video and process instructions
while current_step < len(instructions):
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Display instruction text on the video frame
    cv2.putText(frame, instructions[current_step], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if processing_step:
        print(instructions[current_step])
        time.sleep(2)  # Give the user time to adjust

    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

    if face_landmarks_list:
        for face_landmarks in face_landmarks_list:
            if steps[current_step] == "left":
                # Check if nose bridge landmarks are detected
                if 'nose_bridge' in face_landmarks:
                    nose_bridge = face_landmarks['nose_bridge']
                    print("Left side nose bridge points:", nose_bridge)
                    if len(nose_bridge) >= 2 and nose_bridge[0][0] < nose_bridge[1][0]:
                        capture_image(frame, steps[current_step])
                        current_step += 1
            elif steps[current_step] == "right":
                # Check if nose bridge landmarks are detected
                if 'nose_bridge' in face_landmarks:
                    nose_bridge = face_landmarks['nose_bridge']
                    print("Right side nose bridge points:", nose_bridge)
                    if len(nose_bridge) >= 2 and nose_bridge[0][0] > nose_bridge[1][0]:
                        capture_image(frame, steps[current_step])
                        current_step += 1
            elif steps[current_step] == "forward":
                # Check if top lip landmarks are detected
                if 'top_lip' in face_landmarks:
                    top_lip = face_landmarks['top_lip']
                    print("Forward facing top lip points:", top_lip)
                    if len(top_lip) >= 7 and top_lip[0][0] < top_lip[6][0]:
                        capture_image(frame, steps[current_step])
                        current_step += 1
                        # Check if face matches reference image
                        face_encoding = face_recognition.face_encodings(rgb_small_frame)
                        if len(face_encoding) > 0:
                            match = face_recognition.compare_faces([reference_face_encoding], face_encoding[0])
                            if match[0]:
                                match_found = True
                                cv2.putText(frame, "Match Found!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                print("Match found with reference image!")
            break

    cv2.imshow('Video', frame)
    processing_step = not processing_step

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
