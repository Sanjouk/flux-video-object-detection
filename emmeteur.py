import socket
import cv2
import struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5000))
server_socket.listen(1)

conn, addr = server_socket.accept()
camera = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Compression en JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()

        # Envoi de la taille (4 octets, entier grand-boutiste) puis du buffer
        size = struct.pack('>I', len(data))
        conn.sendall(size + data)

finally:
    camera.release()
    conn.close()
    server_socket.close()