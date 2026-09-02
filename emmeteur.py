import cv2
import socket

# Configuration du récepteur
RECEIVER_IP = '10.77.180.142'  # Remplace par l'IP de la machine qui fait le traitement YOLO
PORT = 5000
MAX_UDP_SIZE = 65507  # Taille maximale du payload d'un paquet UDP

# Initialisation de la socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Ouverture de la webcam (0 = caméra intégrée/USB par défaut)
cap = cv2.VideoCapture(0)

# Réduction de la résolution pour limiter la bande passante et la taille des paquets
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Compression en JPEG (qualité 50 % pour garantir que l'image reste légère)
    encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    if not encoded:
        continue

    data = buffer.tobytes()

    # Vérification que l'image ne dépasse pas la taille maximale autorisée en UDP
    if len(data) > MAX_UDP_SIZE:
        print(f"Avertissement : Frame trop lourde ({len(data)} octets), ignorée.")
        continue

    # Envoi direct du datagramme au récepteur
    try:
        sock.sendto(data, (RECEIVER_IP, PORT))
    except Exception as e:
        print(f"Erreur lors de l'envoi : {e}")

cap.release()
sock.close()