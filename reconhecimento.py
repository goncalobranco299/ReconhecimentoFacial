""" import cv2
import mediapipe as mp
import os
import numpy as np
from dataset_360 import DATASET_DIR


# Pasta com datasets de pessoas
DATASET_DIR = "dataset_360"

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Listas para embeddings e nomes
known_face_embeddings = []
known_face_names = []

# Função para extrair embedding do rosto
def get_face_embedding(image, face_mesh):
    if image is None:
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        embeddings = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            embedding = np.array([[l.x, l.y, l.z] for l in landmarks]).flatten()
            embeddings.append(embedding)
        return embeddings
    return []

# Carregar embeddings do dataset
with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Não foi possível carregar {img_path}")
                continue

            embeddings = get_face_embedding(img, face_mesh)
            for embedding in embeddings:
                known_face_embeddings.append(embedding)
                known_face_names.append(person_name)

print(f"[INFO] {len(known_face_names)} rostos carregados.")

# Abrir webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Processamento em tempo real
with mp_face_mesh.FaceMesh(
        max_num_faces=10,  # Permitir múltiplos rostos
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Não foi possível capturar o frame da câmera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                embedding = np.array([[l.x, l.y, l.z] for l in face_landmarks.landmark]).flatten()

                name = "Desconhecido"

                # Comparar com embeddings conhecidos
                if known_face_embeddings:
                    distances = [np.linalg.norm(embedding - k) for k in known_face_embeddings]
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < 2.5:  # Threshold
                        name = known_face_names[best_match_index]

                # Desenhar landmarks do rosto
                mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )

                # Mostrar nome na tela próximo ao rosto
                h, w, _ = frame.shape
                x = int(face_landmarks.landmark[1].x * w)
                y = int(face_landmarks.landmark[1].y * h) - 10
                cv2.putText(frame, name, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import os
import numpy as np

# Pasta com datasets de pessoas
DATASET_DIR = "dataset_360"

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Listas para embeddings e nomes
known_face_embeddings = []
known_face_names = []

# Função para extrair embedding do rosto
def get_face_embedding(image, face_mesh):
    if image is None:
        return []
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        embeddings = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            embedding = np.array([[l.x, l.y, l.z] for l in landmarks]).flatten()
            embeddings.append(embedding)
        return embeddings
    return []

# Carregar embeddings do dataset
print("[INFO] Carregando dataset...")
with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Não foi possível carregar {img_path}")
                continue

            embeddings = get_face_embedding(img, face_mesh)
            for embedding in embeddings:
                # Evitar duplicação do mesmo rosto
                if len(known_face_embeddings) == 0 or all(np.linalg.norm(embedding - k) > 1.0 for k in known_face_embeddings):
                    known_face_embeddings.append(embedding)
                    known_face_names.append(person_name)

print(f"[INFO] {len(known_face_names)} rostos carregados.")

# Abrir webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Processamento em tempo real
with mp_face_mesh.FaceMesh(
        max_num_faces=10,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Não foi possível capturar o frame da câmera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                embedding = np.array([[l.x, l.y, l.z] for l in face_landmarks.landmark]).flatten()

                name = "Desconhecido"

                # Comparar com embeddings conhecidos
                if known_face_embeddings:
                    distances = [np.linalg.norm(embedding - k) for k in known_face_embeddings]
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < 2.5:
                        name = known_face_names[best_match_index]

                # Desenhar landmarks do rosto
                mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )

                # Mostrar nome próximo ao rosto
                h, w, _ = frame.shape
                x = int(face_landmarks.landmark[1].x * w)
                y = int(face_landmarks.landmark[1].y * h) - 10
                cv2.putText(frame, name, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

 """