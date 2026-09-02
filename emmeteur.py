import cv2
import socket

# 1. Utilise l'IP directe pour tester d'abord si le flux passe
TARGET_IP = '10.180.183.191'  # Ou '10.180.183.255' pour le broadcast sous-réseau
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

cap = cv2.VideoCapture(0)

# Réduction de la résolution à 480x360 pour alléger drastiquement les paquets UDP
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print(f"Début de l'envoi vers {TARGET_IP}:{PORT}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Qualité 40% pour garantir un paquet sous les 15-20 Ko
    encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    if not encoded:
        continue

    data = buffer.tobytes()

    try:
        sock.sendto(data, (TARGET_IP, PORT))
    except Exception as e:
        print(f"Erreur d'envoi : {e}")

cap.release()
sock.close()