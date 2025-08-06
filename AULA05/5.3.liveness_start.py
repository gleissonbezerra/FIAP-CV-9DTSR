import cv2
import mediapipe as mp
import numpy as np
import time
from math import fabs

def eye_aspect_ratio(landmarks, eye_indices):

    # pontos relevantes
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    
    # fórmula EAR (EYE ASPECT RATIO)
    A = np.linalg.norm(p2 - p4)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p6)
    ear = (A + B) / 2.0 * C 

    return ear


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

DELTA_THRESHOLD = .00001
CONSEC_FRAMES = 3
blink_counter = 0
blink_total = 0

cap = cv2.VideoCapture(0)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
previous_ear_avg = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ## --- COMPLETE AQUI --- ##

    cv2.imshow("Prova de Vida Ativa", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

    time.sleep(.1)


cap.release()
cv2.destroyAllWindows()
