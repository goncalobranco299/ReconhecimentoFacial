import cv2
import mediapipe as mp
import os


nome = input("Digite o nome da pessoa: ")
if nome == "":
    print("Nome inválido.")
    exit()


DATASET_DIR = os.path.join("dataset_360", nome)
os.makedirs(DATASET_DIR, exist_ok=True)



mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
total_images = 250

print(f"\nCapturando 360° do rosto de: {nome}")
print("Gire o rosto lentamente...\n")



while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb)

    message = ""

    if result.detections:
        det = result.detections[0]
        bbox = det.location_data.relative_bounding_box

        # Coordenadas 
        h, w, _ = frame.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Ajustes simples para melhor corte
        x = max(0, x)
        y = max(0, y)
        bw = min(w - x, bw)
        bh = min(h - y, bh)

        if bw < 120 or bh < 120:
            message = "Chegue mais perto"
        
    
        elif bw > 400 or bh > 400:
            message = "Afastar um pouco"
        
        # c) rosto deve estar centralizado
        cx = x + bw // 2
        cy = y + bh // 2

        if cx < w*0.25 or cx > w*0.75:
            message = "Centralize o rosto"
        elif cy < h*0.25 or cy > h*0.75:
            message = "Centralize verticalmente"

        else:
            # Tudo OK → salvar
            face_img = frame[y:y+bh, x:x+bw]

            cv2.imwrite(f"{DATASET_DIR}/face_{count:03d}.jpg", face_img)
            count += 1
            message = f"Capturando {count}/{total_images}"

        # Mostrar caixa ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0,255,0), 2)

    else:
        message = "Nenhum rosto detectado"

    cv2.putText(frame, message, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("360 Face Capture", frame)

    if count >= total_images:
        print("Captura completa!")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Cancelado pelo usuário.")
        break


cap.release()
cv2.destroyAllWindows()
