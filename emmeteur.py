import socket
import cv2
import struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5000))
server_socket.listen(1)

conn, addr = server_socket.accept()
camera = cv2.VideoCapture(0)

try:
    if not camera.isOpened():
        print("Erreur : Impossible d'accéder à la webcam")
        exit()
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # 1. Redimensionner l'image (ex: 480x360 au lieu de 1080p/720p)
        frame = cv2.resize(frame, (480, 360))

        # 2. Compresser le JPEG à 50% de qualité (au lieu de 95% par défaut)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        data = buffer.tobytes()

        # Envoi de la taille (4 octets, entier grand-boutiste) puis du buffer
        size = struct.pack('>I', len(data))
        conn.sendall(size + data)

finally:
    camera.release()
    conn.close()
    server_socket.close()