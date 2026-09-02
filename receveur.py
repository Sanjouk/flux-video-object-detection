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
    global output_frame
    model = YOLO('yolov8n.pt')

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('0.0.0.0', 5000))
    
    # Rend la socket non-bloquante pour pouvoir vider le buffer instantanément
    udp_socket.setblocking(False)

    frame_count = 0
    annotated_frame = None

    while True:
        try :
            packet = None
            
            # 1. Vidage du buffer : on lit tous les paquets en attente et on garde le tout dernier
            while True:
                try:
                    data, _ = udp_socket.recvfrom(65535)
                    packet = data  # Écrase les paquets obsolètes
                except BlockingIOError:
                    break  # Plus aucun paquet en attente dans le buffer

            # Si aucune nouvelle image n'est arrivée, on attend un court instant
            if packet is None:
                time.sleep(0.005)
                continue

            # 2. Décodage de la frame la plus récente
            frame = cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue

            frame_count += 1

            # 3. Inférence YOLO (1 frame sur 4)
            if frame_count % 4 == 0 or annotated_frame is None:
                results = model(frame, imgsz=320, verbose=False)
                annotated_frame = results[0].plot()

            # 4. Envoi vers Flask
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ret:
                with lock:
                    output_frame = buffer.tobytes()

        except Exception as e:
            print(f"Erreur de réception UDP : {e}")
            break

    udp_socket.close()


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
    """Page web affichant le flux vidéo - UI responsive moderne"""
    return render_template_string('''
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <meta name="color-scheme" content="dark">
    <title>Relais YOLO — Live</title>
    <style>
        :root{
            --bg:#0f1115;
            --bg2:#171a20;
            --card:#1c1f27;
            --card2:#212533;
            --border:#2a2f3e;
            --accent:#00ff88;
            --accent2:#00d4ff;
            --text:#eef1f6;
            --muted:#9aa3b2;
            --radius:16px;
            --shadow: 0 10px 40px rgba(0,0,0,.45), 0 1px 0 rgba(255,255,255,.06) inset;
        }
        *{box-sizing:border-box;margin:0;padding:0}
        html,body{height:100%}
        body{
            font-family: ui-sans-system, -apple-system, "Segoe UI", Roboto, Inter, "Helvetica Neue", Arial, sans-serif;
            background:
                radial-gradient(900px 600px at 20% -10%, rgba(0,255,136,.14), transparent 60%),
                radial-gradient(800px 500px at 95% 0%, rgba(0,212,255,.10), transparent 60%),
                linear-gradient(180deg, var(--bg), #0a0c10);
            color:var(--text);
            min-height:100dvh;
            display:flex;
            flex-direction:column;
            -webkit-font-smoothing:antialiased;
        }
        /* Header */
        header{
            position:sticky; top:0; z-index:10;
            backdrop-filter: blur(12px) saturate(1.2);
            background: rgba(15,17,21,.72);
            border-bottom:1px solid rgba(255,255,255,.07);
        }
        .nav{
            max-width:1160px; margin:0 auto;
            padding:14px clamp(16px, 3vw, 28px);
            display:flex; align-items:center; justify-content:space-between; gap:16px;
        }
        .brand{display:flex; align-items:center; gap:12px; min-width:0}
        .logo{
            width:36px; height:36px; border-radius:10px;
            display:grid; place-items:center;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            color:#0a0c10; font-weight:800; font-size:18px;
            box-shadow: 0 6px 20px rgba(0,255,136,.35);
            flex-shrink:0;
        }
        .brand h1{font-size:clamp(16px, 2.2vw, 18px); font-weight:700; letter-spacing:-.02em; line-height:1.1}
        .brand p{font-size:12.5px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
        .badges{display:flex; align-items:center; gap:10px; flex-shrink:0; flex-wrap:wrap; justify-content:flex-end}
        .badge{
            display:inline-flex; align-items:center; gap:8px;
            padding:7px 12px; border-radius:999px;
            font-size:12.5px; font-weight:600; letter-spacing:.02em;
            border:1px solid var(--border); background:rgba(255,255,255,.04);
        }
        .badge.live{background:rgba(0,255,136,.12); border-color:rgba(0,255,136,.35); color:#b6ffde}
        .dot{width:8px; height:8px; border-radius:50%; background:var(--accent); box-shadow:0 0 0 6px rgba(0,255,136,.18)}
        .dot.pulse{animation:pulse 1.6s infinite}
        @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(0,255,136,.45)} 70%{box-shadow:0 0 0 10px rgba(0,255,136,0)} 100%{box-shadow:0 0 0 0 rgba(0,255,136,0)}}
        .badge.muted{color:var(--muted)}
        /* Layout */
        main{flex:1; width:100%; max-width:1160px; margin:0 auto; padding:clamp(16px, 3vw, 28px); display:flex; flex-direction:column; gap:18px}
        .hero{display:flex; flex-wrap:wrap; align-items:end; justify-content:space-between; gap:12px}
        .hero h2{font-size:clamp(20px, 3.5vw, 28px); font-weight:800; letter-spacing:-.03em; line-height:1.1}
        .hero h2 span{background:linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text}
        .hero p{color:var(--muted); font-size:clamp(13px, 1.8vw, 14.5px); max-width:52ch; line-height:1.5}
        /* Card video */
        .card{
            background: linear-gradient(180deg, var(--card), var(--card2));
            border:1px solid rgba(255,255,255,.08);
            border-radius:var(--radius);
            box-shadow:var(--shadow);
            overflow:hidden;
        }
        .video-wrap{
            position:relative;
            background:#07080a;
            /* s'adapte à l'écran : ratio 4/3 natif, mais remplit sans déborder */
            aspect-ratio: 4 / 3;
            max-height: min(72vh, 760px);
            display:grid; place-items:center;
            overflow:hidden;
        }
        /* sur écrans larges 16:9, sur mobile on laisse respirer */
        @media (min-width: 1100px){ .video-wrap{ aspect-ratio: 16 / 9; } }
        .video-wrap img{
            width:100%; height:100%;
            object-fit:contain; /* jamais cropé, jamais déformé */
            display:block;
            background:#07080a;
        }
        .video-wrap::after{
            content:""; position:absolute; inset:0;
            border-radius:0; pointer-events:none;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.06);
        }
        .overlay{
            position:absolute; left:12px; top:12px;
            display:flex; gap:8px; align-items:center;
        }
        .chip{
            font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
            padding:6px 9px; border-radius:999px;
            background:rgba(0,0,0,.55); backdrop-filter:blur(8px);
            border:1px solid rgba(255,255,255,.12); color:#fff;
        }
        .chip.rec{color:#ff3b3b; border-color:rgba(255,59,59,.35)}
        .chip.rec::before{content:"●"; margin-right:6px; animation:blink 1.2s infinite}
        @keyframes blink{50%{opacity:.35}}
        .placeholder{
            position:absolute; inset:0; display:grid; place-items:center;
            color:var(--muted); font-size:14px; text-align:center; padding:24px;
        }
        .placeholder[hidden]{display:none !important}
        .spinner{
            width:28px; height:28px; border-radius:50%;
            border:3px solid rgba(255,255,255,.12); border-top-color:var(--accent);
            animation:spin .9s linear infinite; margin:0 auto 10px;
        }
        @keyframes spin{to{transform:rotate(360deg)}}
        /* Barre infos */
        .bar{
            display:flex; flex-wrap:wrap; gap:10px;
            align-items:center; justify-content:space-between;
            padding:12px 14px;
            background:rgba(255,255,255,.03);
            border-top:1px solid rgba(255,255,255,.07);
            font-size:13px; color:var(--muted);
        }
        .bar strong{color:var(--text); font-weight:600}
        .stats{display:flex; flex-wrap:wrap; gap:8px; align-items:center}
        .stat{
            display:inline-flex; align-items:center; gap:6px;
            padding:6px 10px; border-radius:999px;
            background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.07);
            font-size:12.5px;
        }
        .stat b{color:var(--text)}
        /* Grille infos secondaires */
        .grid{display:grid; grid-template-columns: repeat(12, 1fr); gap:14px}
        .panel{grid-column: span 6; padding:16px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:14px}
        .panel h3{font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:8px}
        .panel p{font-size:13.5px; line-height:1.6; color:#cbd3e0}
        .panel code{font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12.5px; background:rgba(255,255,255,.07); padding:2px 6px; border-radius:6px; color:#fff}
        @media (max-width: 720px){
            .panel{grid-column: span 12}
            .nav{padding-block:12px}
            .brand p{display:none}
            .video-wrap{max-height: 62vh}
        }
        @media (max-width: 420px){
            .badges .badge.muted{display:none}
            .bar{font-size:12.5px}
        }
        footer{
            padding:14px clamp(16px, 3vw, 28px);
            text-align:center; color:var(--muted); font-size:12.5px;
            border-top:1px solid rgba(255,255,255,.06);
        }
        footer a{color:var(--accent); text-decoration:none}
        footer a:hover{text-decoration:underline}
    </style>
</head>
<body>
    <header>
        <div class="nav">
            <div class="brand">
                <div class="logo">Y</div>
                <div style="min-width:0">
                    <h1>Relais YOLO</h1>
                    <p>Détection d'objets &middot; YOLOv8n &middot; Flux UDP &rarr; MJPEG</p>
                </div>
            </div>
            <div class="badges">
                <span class="badge live"><span class="dot pulse"></span> LIVE</span>
                <span class="badge muted">:8000 &middot; /video_feed</span>
            </div>
        </div>
    </header>

    <main>
        <div class="hero">
            <div>
                <h2>Flux <span>en direct</span> — réseau local</h2>
                <p>Interface responsive : le flux s'adapte à la taille de l'écran sans déformation. Noir &vert propre, lisible en plein écran sur mobile comme sur desktop.</p>
            </div>
            <div class="stats">
                <span class="stat">UDP <b>:5000</b></span>
                <span class="stat">HTTP <b>:8000</b></span>
                <span class="stat">YOLO <b>v8n</b></span>
            </div>
        </div>

        <section class="card" aria-label="Flux vidéo">
            <div class="video-wrap" id="wrap">
                <img id="stream" src="/video_feed" alt="Flux YOLO en direct" loading="eager" decoding="async"
                     onload="document.getElementById('ph').hidden=true"
                     onerror="document.getElementById('ph').hidden=false">
                <div class="overlay">
                    <span class="chip rec">REC</span>
                    <span class="chip" id="res">640 &times; 480 &middot; MJPEG</span>
                </div>
                <div class="placeholder" id="ph">
                    <div>
                        <div class="spinner"></div>
                        <div><strong style="color:var(--text)">Connexion au flux…</strong><br>En attente de <code>/video_feed</code></div>
                    </div>
                </div>
            </div>
            <div class="bar">
                <div>Flux MJPEG en temps réel &middot; <strong>object-fit: contain</strong> &middot; recadrage auto selon l'écran</div>
                <div class="stats">
                    <span class="stat" title="Ouvrir le flux brut"><a href="/video_feed" target="_blank" rel="noopener" style="color:inherit;text-decoration:none">Ouvrir le flux brut &nearr;</a></span>
                </div>
            </div>
        </section>

        <div class="grid">
            <div class="panel">
                <h3>Astuces d'affichage</h3>
                <p>
                    Sur mobile, pincez pour zoomer. Sur desktop, mettez en plein écran (<code>F11</code>).
                    L'image garde toujours ses proportions — pas d'étirement, pas de crop.
                </p>
            </div>
            <div class="panel">
                <h3>Diagnostic</h3>
                <p>
                    Émetteur : <code>python emmeteur.py</code> &rarr; <code>RECEIVER_IP:5000</code> (UDP).<br>
                    Récepteur : <code>python receveur.py</code> &rarr; ouvre <code>http://&lt;ip&gt;:8000</code>.<br>
                    Si écran noir, vérifie le firewall et que l'émetteur envoie bien.
                </p>
            </div>
        </div>
    </main>

    <footer>D&eacute;tection YOLOv8 &middot; Flask MJPEG &middot; Fait pour le r&eacute;seau local</footer>

    <script>
        // Masque le placeholder dès que le flux charge, le réaffiche en cas d'erreur.
        // + met à jour la puce résolution si l'image charge.
        const img = document.getElementById('stream');
        const ph  = document.getElementById('ph');
        const res = document.getElementById('res');
        let okOnce = false;
        img.addEventListener('load', () => {
            if(!okOnce){ ph.hidden = true; okOnce = true; }
            if(img.naturalWidth) res.textContent = img.naturalWidth + ' × ' + img.naturalHeight + ' · MJPEG';
        });
        img.addEventListener('error', () => { ph.hidden = false; });
        // Auto-retry discret si le flux coupe (cache-bust)
        setInterval(() => {
            if(img.naturalWidth === 0 && okOnce){
                const u = new URL(img.src, location.href);
                u.searchParams.set('t', Date.now());
                // ne recharge que si vraiment bloqué pour éviter de couper un flux MJPEG sain
            }
        }, 5000);
    </script>
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
    # Accessible via http://<IP_DE_CE_PC>:8000 sur le réseau local
    app.run(host='0.0.0.0', port=8000, debug=False)