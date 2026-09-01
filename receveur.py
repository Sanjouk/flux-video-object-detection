import socket
import cv2
import numpy as np
import struct

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 5000))

data = b""
payload_size = struct.calcsize('>I')

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
    cv2.imshow('Flux TCP Brut', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()