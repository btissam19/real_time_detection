import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Initialize InceptionResnetV1 for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load known faces and their names
known_faces = []  # List to store known face embeddings
known_names = []  # List to store names corresponding to known faces

# Function to add a new known face
def add_known_face(img, name):
    faces = mtcnn(img)
    if faces is not None and len(faces) > 0:
        face = faces[0]
        face = face.unsqueeze(0)  # Add batch dimension
        embedding = resnet(face).detach().cpu()
        known_faces.append(embedding)
        known_names.append(name)

# Load an image with a known face
img = cv2.imread('reference.jpg')
add_known_face(img, 'ibtissam')

# Function to recognize faces
def recognize_face(face):
    embedding = resnet(face.unsqueeze(0)).detach().cpu()
    distances = [torch.dist(embedding, known_face).item() for known_face in known_faces]
    min_dist = min(distances)
    if min_dist < 0.6:  # Threshold for recognizing a face
        return known_names[distances.index(min_dist)]
    return "Unknown"

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces, _ = mtcnn.detect(frame)
    
    if faces is not None:
        for face in faces:
            x1, y1, x2, y2 = [int(coord) for coord in face]
            face_img = frame[y1:y2, x1:x2]
            
            # Convert face image to RGB
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_tensor = torch.tensor(face_img_rgb).permute(2, 0, 1).float() / 255.0
            
            # Recognize face
            name = recognize_face(face_img_tensor)
            
            # Draw bounding box and name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
