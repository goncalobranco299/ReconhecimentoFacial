import cv2
import mediapipe as mp
import os
import numpy as np
from collections import deque

DATASET_DIR = "dataset_360"

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils



def extract_embedding(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

    pts -= np.mean(pts)

    norm = np.linalg.norm(pts)
    if norm > 0:
        pts /= norm

    return pts


def face_quality(landmarks):
    """
    Retorna score 0–1 que indica se:
    - a face está completa
    - não está muito virada
    - não está comprimida
    """

    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    dx = right_eye.x - left_eye.x
    dy = abs(nose.x - (left_eye.x + right_eye.x) / 2)

    yaw = dy / abs(dx + 1e-6)  

    score = 1.0 - min(yaw, 1.0)

    return score  


def distance(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_dist = 1 - cos
    l2 = np.linalg.norm(a - b)
    return cos_dist * 0.6 + l2 * 0.4

print("[INFO] Carregando dataset...")

known_face_embeddings = {}
known_face_names = []
auto_thresholds = {}

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
    for person_name in os.listdir(DATASET_DIR):

        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []

        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            lm = results.multi_face_landmarks[0].landmark

            if face_quality(lm) < 0.35:
                continue

            emb = extract_embedding(lm)
            embeddings.append(emb)

        if len(embeddings) < 3:
            print(f"[AVISO] Poucas imagens válidas para {person_name}")
            continue

        embeddings = np.array(embeddings)
        mean_emb = np.mean(embeddings, axis=0)
        dists = np.linalg.norm(embeddings - mean_emb, axis=1)

        good_idx = dists < np.percentile(dists, 80)  
        embeddings = embeddings[good_idx]

        final_emb = np.mean(embeddings, axis=0)
        known_face_embeddings[person_name] = final_emb
        known_face_names.append(person_name)

        auto_thresholds[person_name] = np.mean(dists)

print(f"[INFO] {len(known_face_names)} pessoas registradas.")


cap = cv2.VideoCapture(0)

smooth_window = 5
pred_queue = deque(maxlen=smooth_window)

with mp_face_mesh.FaceMesh(max_num_faces=3,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.6) as face_mesh:

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for lm in results.multi_face_landmarks:

                if face_quality(lm.landmark) < 0.40:
                    label = "Muito Virado"
                else:
                    emb = extract_embedding(lm.landmark)

                    best_name = "Desconhecido"
                    best_dist = 999

                    for person_name, saved_emb in known_face_embeddings.items():
                        dist = distance(emb, saved_emb)

                        if dist < best_dist:
                            best_dist = dist
                            best_name = person_name

                    person_threshold = auto_thresholds.get(best_name, 1.1) * 1.4

                    if best_dist > person_threshold:
                        best_name = "Desconhecido"

                    pred_queue.append(best_name)
                    label = max(set(pred_queue), key=pred_queue.count)

                mp_drawing.draw_landmarks(
                    frame, lm,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    None,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

                h, w, _ = frame.shape
                cx = int(lm.landmark[1].x * w)
                cy = int(lm.landmark[1].y * h) - 10

                cv2.putText(frame, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento Facial Otimizado", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
