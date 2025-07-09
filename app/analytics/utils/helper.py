import cv2
import numpy as np 
import dlib
from tqdm import tqdm
import os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def extract_face(image, net, predictor):    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Convert bounding box to dlib rectangle format
            dlib_rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmarks = predictor(gray, dlib_rect)  
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
            x, y, w, h = cv2.boundingRect(landmarks_np)
            x -= 25
            y -= 25
            w += 50
            h += 50

            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            face_crop=image[y:y+h,x:x+w]
            # Crop and resize the face
            try:
                face_crop = cv2.resize(face_crop, (224, 224))
            except:
                face_crop = cv2.resize(image, (224, 224))
            return face_crop,landmarks_np,(w,h)
    return None,None,None

def extract_faces_from_frames(frames, net, predictor):
    faces_list = []
    landmarks_list = []
    sizes_list = []

    for image in tqdm(frames):
        face_crop, landmarks_np, size = extract_face(image, net, predictor)
        
        # Append the results to the respective lists
        faces_list.append(face_crop)
        landmarks_list.append(landmarks_np)
        sizes_list.append(size)

    return faces_list, landmarks_list, sizes_list

def plot_facial_expression_graphs(smile_data, ear_data, yawn_data, thresholds, full_path:str):
    """
    Plots multiple subplots (smile, EAR, and yawn ratios) in one figure.
    
    Parameters:
    - smile_data: List of smile ratios.
    - ear_data: List of eye aspect ratios (EAR).
    - yawn_data: List of yawn ratios.
    - thresholds: List containing thresholds for smile, EAR, and yawn.
    - path: Path to save the combined plot.
    
    Returns:
    - buf: BytesIO buffer containing the saved plot.
    """
    plt.figure(figsize=(12, 8))  # Create a figure of appropriate size

    # Plot smile data
    plt.subplot(3, 1, 1)
    plt.plot(smile_data, label='Smile Ratio (Width/Face Width)')
    plt.axhline(y=thresholds[0], color='black', linestyle='--', label='Threshold')
    plt.title('Smile Ratio Over Time')
    plt.ylabel('Ratio')
    plt.legend()

    # Plot EAR data
    plt.subplot(3, 1, 2)
    plt.plot(ear_data, label='Eye Aspect Ratio (EAR)', color='orange')
    plt.axhline(y=thresholds[1], color='black', linestyle='--', label='Threshold')
    plt.title('Eye Aspect Ratio (EAR) Over Time')
    plt.ylabel('Ratio')
    plt.legend()

    # Plot yawn data
    plt.subplot(3, 1, 3)
    plt.plot(yawn_data, label='Yawn Ratio (Mouth Height/Face Height)', color='red')
    plt.axhline(y=thresholds[2], color='black', linestyle='--', label='Threshold')
    plt.title('Yawn Ratio Over Time')
    plt.xlabel('Frames')
    plt.ylabel('Ratio')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(full_path, format='png')
    plt.clf()  # Clear the figure after saving
    return full_path

def annotate_vid(frames,annotations1,annotations2,annotations3,output_video_path):
    # Output video path

    # Define video parameters
    fps = 30  # Adjust FPS as per your requirement
    frame_height, frame_width = frames[0].shape[:2]

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Annotate each frame and write it to the output video
    for i, frame in enumerate(frames):
        frame=np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get annotations for the current frame
        annotation1 = annotations1[i]
        annotation2 = annotations2[i]
        annotation3 = annotations3[i]
        
        # Add text annotations on the frame
        cv2.putText(frame, f'Blink : {annotation1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Yawn : {annotation2}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Smile : {annotation3}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Write the annotated frame to the output video
        out.write(frame)

    # Release resources
    out.release()
    cv2.destroyAllWindows()
