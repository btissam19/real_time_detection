from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import math
import requests
import argparse
import torch
import cv2

parser = argparse.ArgumentParser("Face pose detection for one face")
parser.add_argument("-r", "--ref", help="Path to the reference image.", type=str, required=True)
args = parser.parse_args()

ref_path = args.ref

left_offset = 20
fontScale = 2
fontThickness = 3
text_color = (0,0,255)
lineColor = (255, 255, 0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

mtcnn = MTCNN(image_size=160,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], # MTCNN thresholds
              factor=0.709,
              post_process=True,
              device=device # If you don't have GPU
        )

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the reference image and calculate its embedding
def get_face_embedding(img_path):
    img = Image.open(img_path)
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
        return img_embedding
    return None

ref_embedding = get_face_embedding(ref_path)

# Landmarks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]
def npAngle(a, b, c):
    ba = a - b
    bc = c - b 
    
    cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def visualize(image, landmarks_, angle_R_, angle_L_, pred_):
    fig , ax = plt.subplots(1, 1, figsize= (8,8))
    
    leftCount = len([i for i in pred_ if i == 'Left Profile'])
    rightCount = len([i for i in pred_ if i == 'Right Profile'])
    frontalCount = len([i for i in pred_ if i == 'Frontal'])
    facesCount = len(pred_) # Number of detected faces (above the threshold)
    ax.set_title(f"Number of detected faces = {facesCount} \n frontal = {frontalCount}, left = {leftCount}, right = {rightCount}")
    for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
        
        if pred == 'Frontal':
            color = 'white'
        elif pred == 'Right Profile':
            color = 'blue'
        else:
            color = 'red'
            
        point1 = [landmarks[0][0], landmarks[1][0]]
        point2 = [landmarks[0][1], landmarks[1][1]]

        point3 = [landmarks[2][0], landmarks[0][0]]
        point4 = [landmarks[2][1], landmarks[0][1]]

        point5 = [landmarks[2][0], landmarks[1][0]]
        point6 = [landmarks[2][1], landmarks[1][1]]
        for land in landmarks:
            ax.scatter(land[0], land[1])
        plt.plot(point1, point2, 'y', linewidth=3)
        plt.plot(point3, point4, 'y', linewidth=3)
        plt.plot(point5, point6, 'y', linewidth=3)
        plt.text(point1[0], point2[0], f"{pred} \n {math.floor(angle_L)}, {math.floor(angle_R)}", 
                size=20, ha="center", va="center", color=color)
        ax.imshow(image)
        fig.savefig('Output_detection.jpg')
    return print('Done detect')

def visualizeCV2(frame, landmarks_, angle_R_, angle_L_, pred_):
    
    for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
        
        if pred == 'Frontal':
            color = (0, 0, 0)
        elif pred == 'Right Profile':
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
            
        point1 = [int(landmarks[0][0]), int(landmarks[1][0])]
        point2 = [int(landmarks[0][1]), int(landmarks[1][1])]

        point3 = [int(landmarks[2][0]), int(landmarks[0][0])]
        point4 = [int(landmarks[2][1]), int(landmarks[0][1])]

        point5 = [int(landmarks[2][0]), int(landmarks[1][0])]
        point6 = [int(landmarks[2][1]), int(landmarks[1][1])]

        for land in landmarks:
            cv2.circle(frame, (int(land[0]), int(land[1])), radius=5, color=(0, 255, 255), thickness=-1)
        cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[1][0]), int(landmarks[1][1])), lineColor, 3)
        cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
        cv2.line(frame, (int(landmarks[1][0]), int(landmarks[1][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
        
        text_sizeR, _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_PLAIN, fontScale, 4)
        text_wR, text_hR = text_sizeR
        
        cv2.putText(frame, pred, (point1[0], point2[0]), cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness, cv2.LINE_AA)

def predFacePose(frame):
    bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True) # The detection part producing bounding box, probability of the detected face, and the facial landmarks
    
    if bbox_ is None or prob_ is None or landmarks_ is None:
        print("Detection failed: bbox_, prob_, or landmarks_ is None.")
        return [], [], [], []

    if len(bbox_) == 0 or len(prob_) == 0 or len(landmarks_) == 0:
        print("Detection failed: bbox_, prob_, or landmarks_ is empty.")
        return [], [], [], []

    angle_R_List = []
    angle_L_List = []
    predLabelList = []
    
    for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
        if bbox is not None: # To check if we detect a face in the image
            if prob > 0.9: # To check if the detected face has probability more than 90%, to avoid 
                angR = npAngle(landmarks[0], landmarks[1], landmarks[2]) # Calculate the right eye angle
                angL = npAngle(landmarks[1], landmarks[0], landmarks[2])# Calculate the left eye angle
                angle_R_List.append(angR)
                angle_L_List.append(angL)
                if ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58))):
                    predLabel='Frontal'
                    predLabelList.append(predLabel)
                else: 
                    if angR < angL:
                        predLabel='Left Profile'
                    else:
                        predLabel='Right Profile'
                    predLabelList.append(predLabel)
            else:
                print('The detected face is less than the detection threshold')
        else:
            print('No face detected in the image')
    return landmarks_, angle_R_List, angle_L_List, predLabelList

def predFacePoseApp(ref_embedding):
    source = 0
    # Create a video capture object from the VideoCapture Class.
    video_cap = cv2.VideoCapture(0)
    # Create a named window for the video display.
    win_name = 'Video Preview'
    cv2.namedWindow(win_name)
    video_cadesired_width = 160
    desired_height = 160
    # dim = (desired_width, desired_height)
    left_offset = 20
    fontScale = 2
    fontThickness = 3
    text_color = (0,0,255)
    while True:
        # Read one frame at a time using the video capture object.
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        
        landmarks_, angle_R_List, angle_L_List, predLabelList = predFacePose(frame)

        # Annotate each video frame.
        visualizeCV2(frame, landmarks_, angle_R_List, angle_L_List, predLabelList)
        
        # Perform face recognition
        face_embedding = get_face_embedding_from_frame(frame)
        if face_embedding is not None:
            similarity = (ref_embedding @ face_embedding.T).item()
            match_text = "Match" if similarity > 0.6 else "No Match"
            cv2.putText(frame, f"Similarity: {similarity:.2f} - {match_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1)

        # You can use this feature to check if the user selected the `q` key to quit the video stream.
        if key == ord('Q') or key == ord('q') or key == 27:
            # Exit the loop.
            break

    video_cap.release()
    cv2.destroyWindow(win_name)

def get_face_embedding_from_frame(frame):
    img_cropped = mtcnn(frame)
    if img_cropped is not None:
        img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
        return img_embedding
    return None

if __name__ == '__main__':
    predFacePoseApp(ref_embedding)
