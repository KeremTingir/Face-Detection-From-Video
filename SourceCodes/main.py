import cv2
import numpy as np

# video opening processing
cap = cv2.VideoCapture("Datasets//#####")  # Upload the video path in the datasets folder here

# loading the face recognition model
face_cascade_frontal = cv2.CascadeClassifier("DataSets//haarcascade_frontalface_default.xml")
face_cascade_profile = cv2.CascadeClassifier("DataSets//haarcascade_profileface.xml")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Frontal face detection
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Profile face detection
    faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Check if faces_profile is not empty
    if len(faces_profile) > 0:
        # Concatenate detected faces
        faces = np.vstack((faces_frontal, faces_profile))
    else:
        faces = faces_frontal

    # number of faces
    numFaces = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f'Number of people: {numFaces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
