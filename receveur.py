import socket
import cv2
import numpy as np
import struct
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client_socket.connect(('10.77.180.191', 5000))
client_socket.connect(('10.77.180.142', 5000))

data = b""
payload_size = struct.calcsize('>I')
frame_count = 0
annotated_frame = None

while True:
    # 1. Récupération des 4 octets indiquant la taille
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < payload_size:
        break

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack('>I', packed_msg_size)[0]

    # 2. Récupération de l'image complète selon la taille annoncée
    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # 3. Décodage JPEG et affichage avec OpenCV
    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is not None:
        frame_count += 1

        # N'exécute YOLO que toutes les 3 images
        if frame_count % 3 == 0:
            # imgsz=320 accélère grandement l'inférence
            results = model(frame, imgsz=320, verbose=False)
            annotated_frame = results[0].plot()

            if annotated_frame is not None:
                cv2.imshow('Flux TCP + YOLO', annotated_frame)
            else:
                cv2.imshow('Flux TCP + YOLO', frame)

        # 5. Affichage du résultat
        cv2.imshow('Flux TCP + Détection YOLO', annotated_frame)

    #cv2.imshow('Flux TCP Brut', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()