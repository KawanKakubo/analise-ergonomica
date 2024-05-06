import cv2
import os
import time
import pandas as pd
from datetime import datetime

# Caminho do classificador Haar para detecção de rosto
haarcascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# Inicializa a webcam
cap = cv2.VideoCapture(0)

# Verifica se a webcam está funcionando
if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

# Lista para armazenar dados de postura
posture_data = []

try:
    while True:
        # Captura uma imagem da webcam
        ret, frame = cap.read()
        
        if ret:
            # Converte para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detecta rostos na imagem
            face_cascade = cv2.CascadeClassifier(haarcascade_path)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            timestamp = datetime.now()

            if len(faces) > 0:
                # Obtem o primeiro rosto detectado
                (x, y, w, h) = faces[0]
                center_x = x + w // 2
                center_y = y + h // 2

                # Centro da imagem
                image_center_x = frame.shape[1] // 2
                image_center_y = frame.shape[0] // 2

                threshold = 50  # Tolerância para centralização
                
                # Determina se a postura está correta ou incorreta
                if (abs(center_x - image_center_x) > threshold or
                    abs(center_y - image_center_y) > threshold):
                    status = "Postura Incorreta"
                else:
                    status = "Postura Correta"
            else:
                status = "Inativo"

            # Adiciona à lista
            posture_data.append({'timestamp': timestamp, 'status': status})

        # Aguarda 30 segundos antes da próxima verificação
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Captura interrompida pelo usuário.")

finally:
    # Converte a lista para DataFrame e salva no CSV
    df = pd.DataFrame(posture_data)
    df.to_csv("posture_data.csv", index=False)

    cap.release()
    cv2.destroyAllWindows()
