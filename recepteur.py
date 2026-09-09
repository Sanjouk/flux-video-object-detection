import socket
import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

# État partagé entre le récepteur UDP, YOLO et Flask
output_frame = None
lock = threading.Lock()
latest_frame = None
latest_frame_id = 0
detections = []
output_frame_id = 0

# Gestion dynamique des modèles
ALLOWED_MODELS = {
    'yolo11n.pt': 'YOLO11 Nano (ultra rapide & récent)',
    'yolov8n.pt': 'YOLOv8 Nano (rapide)',
    'yolov8s.pt': 'YOLOv8 Small (équilibré)',
    'yolov8m.pt': 'YOLOv8 Medium (précis)',
}
requested_model_name = 'yolo11n.pt'
active_model_name = 'yolo11n.pt'
is_model_loading = False

UDP_HOST = '0.0.0.0'
UDP_PORT = 5000
MAX_DATAGRAM_SIZE = 65535
JPEG_QUALITY = 60


def draw_detections(frame, current_detections):
    """Dessine les dernières détections sur le frame."""
    for x1, y1, x2, y2, label, confidence in current_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 136), 2)
        text = f'{label} {confidence:.0%}'
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_top = max(y1, text_height + baseline + 6)
        cv2.rectangle(
            frame,
            (x1, label_top - text_height - baseline - 6),
            (x1 + text_width + 8, label_top),
            (0, 255, 136),
            -1,
        )
        cv2.putText(
            frame, text, (x1 + 4, label_top - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 12, 16), 1, cv2.LINE_AA
        )


def receive_frames():
    """Reçoit les frames UDP et vide le buffer pour ne garder que la plus récente."""
    global latest_frame, latest_frame_id, output_frame, output_frame_id

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
    udp_socket.bind((UDP_HOST, UDP_PORT))
    udp_socket.setblocking(False)

    try:
        while True:
            packet = None
            while True:
                try:
                    data, _ = udp_socket.recvfrom(MAX_DATAGRAM_SIZE)
                    packet = data
                except BlockingIOError:
                    break

            if packet is None:
                time.sleep(0.005)
                continue

            try:
                frame = cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    continue

                with lock:
                    latest_frame = frame
                    latest_frame_id += 1
                    current_detections = list(detections)

                web_frame = frame.copy()
                draw_detections(web_frame, current_detections)
                ret, buffer = cv2.imencode(
                    '.jpg', web_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if not ret:
                    continue

                with lock:
                    output_frame = buffer.tobytes()
                    output_frame_id += 1
            except Exception as error:
                app.logger.warning('Frame UDP ignorée : %s', error)
    except OSError as error:
        app.logger.exception('Réception UDP arrêtée : %s', error)
    finally:
        udp_socket.close()


def run_detection(initial_model):
    """Exécute YOLO et bascule de modèle à la volée si demandé."""
    global detections, active_model_name, requested_model_name, is_model_loading

    model = initial_model
    processed_frame_id = 0
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)

    while True:
        # Vérification si un changement de modèle est requis
        if requested_model_name != active_model_name:
            try:
                is_model_loading = True
                print(f"Changement de modèle vers {requested_model_name}...")
                new_model = YOLO(requested_model_name)
                new_model(dummy_frame, imgsz=320, verbose=False)  # Warmup
                model = new_model
                active_model_name = requested_model_name
                print(f"Nouveau modèle {active_model_name} actif !")
            except Exception as e:
                app.logger.error(f"Échec du chargement du modèle {requested_model_name} : {e}")
                requested_model_name = active_model_name
            finally:
                is_model_loading = False

        with lock:
            if latest_frame is None or latest_frame_id == processed_frame_id:
                frame_to_process = None
            else:
                frame_to_process = latest_frame.copy()
                frame_id = latest_frame_id

        if frame_to_process is None:
            time.sleep(0.01)
            continue

        try:
            result = model(frame_to_process, imgsz=320, verbose=False)[0]
            current_detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].item())
                current_detections.append((
                    x1, y1, x2, y2,
                    result.names[class_id],
                    float(box.conf[0].item()),
                ))

            with lock:
                detections = current_detections
        except Exception as error:
            app.logger.exception('Erreur YOLO : %s', error)
        finally:
            processed_frame_id = frame_id


def generate_web_stream():
    """Générateur de flux HTTP MJPEG pour les clients web."""
    last_sent_id = 0
    while True:
        with lock:
            frame_bytes = output_frame
            frame_id = output_frame_id

        if frame_bytes is None or frame_id == last_sent_id:
            time.sleep(0.01)
            continue

        last_sent_id = frame_id
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Page web avec sélecteur de modèle."""
    return render_template_string('''
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="dark">
    <title>Relais YOLO — Live</title>
    <style>
        :root{
            --bg:#0f1115; --card:#1c1f27; --card2:#212533;
            --border:#2a2f3e; --accent:#00ff88; --accent2:#00d4ff;
            --text:#eef1f6; --muted:#9aa3b2; --radius:16px;
        }
        *{box-sizing:border-box;margin:0;padding:0}
        body{
            font-family: system-ui, -apple-system, sans-serif;
            background: linear-gradient(180deg, var(--bg), #0a0c10);
            color:var(--text); min-height:100vh; display:flex; flex-direction:column;
        }
        header{
            position:sticky; top:0; z-index:10; backdrop-filter: blur(12px);
            background: rgba(15,17,21,.75); border-bottom:1px solid rgba(255,255,255,.07);
        }
        .nav{
            max-width:1160px; margin:0 auto; padding:14px 24px;
            display:flex; align-items:center; justify-content:space-between; gap:16px;
        }
        .brand{display:flex; align-items:center; gap:12px;}
        .logo{
            width:36px; height:36px; border-radius:10px; display:grid; place-items:center;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            color:#0a0c10; font-weight:800; font-size:18px;
        }
        .model-picker{
            display:flex; align-items:center; gap:10px; background:rgba(255,255,255,.05);
            padding:6px 12px; border-radius:10px; border:1px solid var(--border);
        }
        .model-picker label{font-size:12.5px; color:var(--muted); font-weight:600;}
        .model-select{
            background:var(--card2); color:var(--text); border:1px solid var(--border);
            padding:6px 10px; border-radius:6px; font-size:13px; outline:none; cursor:pointer;
        }
        .model-select:disabled{opacity:.5; cursor:wait;}
        main{flex:1; width:100%; max-width:1160px; margin:0 auto; padding:24px; display:flex; flex-direction:column; gap:18px;}
        .card{
            background: linear-gradient(180deg, var(--card), var(--card2));
            border:1px solid rgba(255,255,255,.08); border-radius:var(--radius); overflow:hidden;
        }
        .video-wrap{
            position:relative; background:#07080a; aspect-ratio: 4 / 3;
            max-height: min(72vh, 760px); display:grid; place-items:center;
        }
        .video-wrap img{width:100%; height:100%; object-fit:contain; display:block;}
        .status-toast{
            position:absolute; bottom:12px; right:12px; background:rgba(0,0,0,.75);
            padding:6px 12px; border-radius:20px; font-size:12px; border:1px solid var(--border);
            display:none;
        }
    </style>
</head>
<body>
    <header>
        <div class="nav">
            <div class="brand">
                <div class="logo">Y</div>
                <div>
                    <h1 style="font-size:18px">Relais YOLO</h1>
                </div>
            </div>
            
            <div class="model-picker">
                <label for="modelSelect">Modèle :</label>
                <select id="modelSelect" class="model-select" onchange="changeModel(this.value)">
                    {% for file, name in models.items() %}
                        <option value="{{ file }}" {% if file == current_model %}selected{% endif %}>{{ name }}</option>
                    {% endfor %}
                </select>
            </div>
        </div>
    </header>

    <main>
        <section class="card">
            <div class="video-wrap">
                <img id="stream" src="/video_feed" alt="Flux YOLO en direct">
                <div id="toast" class="status-toast">Chargement du modèle...</div>
            </div>
        </section>
    </main>

    <script>
        async function changeModel(modelName) {
            const select = document.getElementById('modelSelect');
            const toast = document.getElementById('toast');
            
            select.disabled = true;
            toast.style.display = 'block';
            toast.textContent = 'Changement de modèle en cours...';

            try {
                const response = await fetch('/set_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: modelName })
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    checkModelStatus(select, toast);
                } else {
                    alert('Erreur : ' + data.message);
                    select.disabled = false;
                    toast.style.display = 'none';
                }
            } catch (err) {
                console.error(err);
                alert('Erreur réseau lors du changement de modèle');
                select.disabled = false;
                toast.style.display = 'none';
            }
        }

        async function checkModelStatus(select, toast) {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch('/model_status');
                    const data = await res.json();
                    if (!data.is_loading) {
                        clearInterval(interval);
                        select.disabled = false;
                        toast.textContent = 'Modèle ' + data.active_model + ' prêt !';
                        setTimeout(() => { toast.style.display = 'none'; }, 2000);
                    }
                } catch (e) {
                    clearInterval(interval);
                    select.disabled = false;
                }
            }, 500);
        }
    </script>
</body>
</html>
    ''', models=ALLOWED_MODELS, current_model=active_model_name)


@app.route('/set_model', methods=['POST'])
def set_model():
    """Endpoint pour demander un changement de modèle."""
    global requested_model_name
    data = request.get_json() or {}
    selected = data.get('model')

    if selected in ALLOWED_MODELS:
        requested_model_name = selected
        return jsonify({'status': 'ok', 'model': selected})
    
    return jsonify({'status': 'error', 'message': 'Modèle non valide'}), 400


@app.route('/model_status')
def model_status():
    """Endpoint de suivi de l'état du modèle."""
    return jsonify({
        'active_model': active_model_name,
        'requested_model': requested_model_name,
        'is_loading': is_model_loading
    })


@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # 1. Préchargement et warmup du modèle par défaut au démarrage
    print(f"Préchargement du modèle {requested_model_name}...")
    is_model_loading = True
    default_model = YOLO(requested_model_name)
    
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    default_model(dummy_frame, imgsz=320, verbose=False)
    
    active_model_name = requested_model_name
    is_model_loading = False
    print(f"Modèle {active_model_name} prêt et préchauffé !")

    # 2. Démarrage des threads avec le modèle préchargé
    threading.Thread(target=receive_frames, daemon=True).start()
    threading.Thread(target=run_detection, args=(default_model,), daemon=True).start()

    # 3. Lancement de Flask
    app.run(host='0.0.0.0', port=8000, debug=False)