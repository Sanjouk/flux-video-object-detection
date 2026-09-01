import socket
import cv2
import numpy as np
import struct
import threading
import time
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Zone mémoire partagée sécurisée entre le thread TCP et le serveur Flask
output_frame = None
lock = threading.Lock()

def receive_and_process():
    """Thread 1 : Reçoit le flux TCP, applique YOLO et met à jour l'image globale"""
    global output_frame

    # 1. Chargement de YOLO
    model = YOLO('yolov8n.pt')

    # 2. Connexion au serveur TCP source (ex: Webcam)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('10.77.180.142', 5000))

    data = b""
    payload_size = struct.calcsize('>I')

    while True:
        try:
            # Récupération de la taille de l'image
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    return
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack('>I', packed_msg_size)[0]

            # Récupération de l'image
            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet:
                    return
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Décodage
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue

            # Inférence YOLO
            results = model(frame, imgsz=320, verbose=False)
            annotated_frame = results[0].plot()

            # Compression en JPEG pour la diffusion Web (qualité 60%)
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ret:
                continue

            # Copie thread-safe de l'image traitée
            with lock:
                output_frame = buffer.tobytes()

        except Exception as e:
            print(f"Erreur de réception TCP : {e}")
            break

    client_socket.close()


def generate_web_stream():
    """Générateur de flux HTTP MJPEG pour les clients web"""
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            frame_bytes = output_frame

        # Envoie l'image au format multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # Limite l'envoi web à ~30 FPS


@app.route('/')
def index():
    """Page web affichant le flux vidéo"""
    return render_template_string('''
        <html>
            <head>
                <title>Relais YOLO - Live Stream</title>
                <style>
                    body { background-color: #1e1e1e; color: white; font-family: sans-serif; text-align: center; margin-top: 20px; }
                    img { border: 2px solid #00ff88; border-radius: 8px; max-width: 90%; height: auto; }
                </style>
            </head>
            <body>
                <h1>Flux YOLO en direct (Réseau Local)</h1>
                <img src="/video_feed">
            </body>
        </html>
    ''')


@app.route('/video_feed')
def video_feed():
    """Endpoint HTTP diffusant le flux vidéo"""
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # 1. Démarrage du récepteur TCP/YOLO en arrière-plan
    t = threading.Thread(target=receive_and_process, daemon=True)
    t.start()

    # 2. Lancement du serveur Web Flask sur le port 8000
    # Accesible via http://<IP_DE_CE_PC>:8000 sur le réseau local
    app.run(host='0.0.0.0', port=8000, debug=False)