import cv2
import mediapipe as mp
import os
import numpy as np

DATASET_DIR = "dataset_360"

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

known_face_embeddings = {}
known_face_names = []



def get_face_embedding(image, face_mesh):
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    embeddings = []
    for face_landmarks in results.multi_face_landmarks:
        pts = np.array([[l.x, l.y, l.z] for l in face_landmarks.landmark]).flatten()

        pts = pts - np.mean(pts)
        norm = np.linalg.norm(pts)
        if norm > 0:
            pts = pts / norm

        
        pts = pts / (np.max(np.abs(pts)) + 1e-6)

        embeddings.append(pts)

    return embeddings



print("[INFO] Carregando dataset...")

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=25) as face_mesh:
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_dir):
            continue

        person_embs = []

        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            embeddings = get_face_embedding(img, face_mesh)
            if embeddings is None:
                continue

            person_embs.extend(embeddings)

        if len(person_embs) > 0:
            known_face_embeddings[person_name] = np.mean(person_embs, axis=0)
            known_face_names.append(person_name)

print(f"[INFO] {len(known_face_names)} pessoas registadas.")



cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

THRESHOLD = 0.5

with mp_face_mesh.FaceMesh(max_num_faces=5,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                embedding = np.array([[l.x, l.y, l.z] for l in face_landmarks.landmark]).flatten()
                embedding = embedding - np.mean(embedding)
                norm = np.linalg.norm(embedding)

                if norm > 0:
                    embedding = embedding / norm

                embedding = embedding / (np.max(np.abs(embedding)) + 1e-6)

                best_name = "Desconhecido"
                best_dist = 999

                for person_name, person_emb in known_face_embeddings.items():
                    dist = np.linalg.norm(embedding - person_emb)

                    if dist < best_dist:
                        best_dist = dist
                        best_name = person_name

                if best_dist > THRESHOLD:
                    best_name = "Desconhecido"

                mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

                h, w, _ = frame.shape
                cx = int(face_landmarks.landmark[1].x * w)
                cy = int(face_landmarks.landmark[1].y * h) - 10
                cv2.putText(frame, best_name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
