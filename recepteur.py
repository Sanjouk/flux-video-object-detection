# recepteur.py : recoit le flux UDP de emeteur.py, detecte avec YOLOv8n, diffuse en MJPEG via Flask.
# Architecture a 3 threads : reception UDP (receive_frames) + inference YOLO (run_detection) + serveur Flask.
# Aucun thread ne bloque les autres : echange via variables globales protegees par `lock`.
import socket
import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

# Serveur web : expose la page '/' et le flux MJPEG '/video_feed'.
app = Flask(__name__)

# Etat partage entre le recepteur UDP, YOLO et Flask.
output_frame = None
lock = threading.Lock()
latest_frame = None
latest_frame_id = 0
detections = []
output_frame_id = 0
# FPS affiches : frames UDP decodees/sec, incrustes sur l'image et exposes via /fps.
fps_display = 0.0
fps_last_update = 0.0
# Compteur d'objets : MAJ par YOLO sous verrou, lu par /fps (meme fraicheur que FPS).
nb_objets = 0

# Config reseau : ecoute sur toutes les interfaces ; 1 datagramme max = 64 Ko - 1 (limite UDP).
# JPEG_QUALITY regle la re-compression web, independante de la qualite d'envoi de l'emetteur.
UDP_HOST = '0.0.0.0'
UDP_PORT = 5000
MAX_DATAGRAM_SIZE = 65535
JPEG_QUALITY = 60


def draw_detections(frame, current_detections):
    """Dessine les dernieres detections sur un nouveau frame."""
    for x1, y1, x2, y2, label, confidence in current_detections:
        # Cadre vert autour de l'objet detecte (epaisseur 2 px).
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 136), 2)
        # Etiquette affichee, ex. "person 87%" : confiance formatee en pourcentage sans decimales.
        text = f'{label} {confidence:.0%}'
        # Mesure le texte pour dimensionner le fond de l'etiquette juste derriere.
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Colle l'etiquette au-dessus de la box, rabattue vers l'interieur si trop pres du haut de l'image.
        label_top = max(y1, text_height + baseline + 6)
        # Fond plein de l'etiquette : garde le texte lisible sur image claire comme sombre.
        cv2.rectangle(
            frame,
            (x1, label_top - text_height - baseline - 6),
            (x1 + text_width + 8, label_top),
            (0, 255, 136),
            -1,
        )
        # Texte sombre antialiase, leger decalage (+4 px) pour ne pas toucher le bord du fond.
        cv2.putText(
            frame, text, (x1 + 4, label_top - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 12, 16), 1, cv2.LINE_AA
        )

def receive_frames():
    """Recoit les frames UDP, vide le buffer pour ne garder que la plus recente et la publie pour le web."""
    global latest_frame, latest_frame_id, output_frame, output_frame_id, fps_display, fps_last_update

    # Socket UDP d'ecoute : recoit les datagrammes JPEG envoyes par emeteur.py.
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Buffer de reception elargi (256 Ko) pour absorber les rafales sans jeter trop de frames.
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Re-bind immediat du port apres un redemarrage.
    udp_socket.bind((UDP_HOST, UDP_PORT))
    udp_socket.setblocking(False)  # Permet le dépilement non-bloquant du buffer

    # Boucle infinie en thread dedie : ne doit jamais bloquer YOLO ni Flask.
    # Compteurs FPS : fenetre glissante d'1 s sur les frames effectivement decodees.
    fps_count = 0
    fps_t0 = time.monotonic()
    try:
        while True:
            packet = None
            
            # Flush du buffer : dépile tous les paquets en attente pour ne garder que le dernier
            while True:
                try:
                    data, _ = udp_socket.recvfrom(MAX_DATAGRAM_SIZE)
                    packet = data
                except BlockingIOError:
                    break

            # Aucun paquet en attente : micro-pause anti-boucle-chaude, puis reessai.
            if packet is None:
                time.sleep(0.005)
                continue

            try:
                # Decode le payload JPEG en image OpenCV (BGR) directement depuis les bytes recus.
                frame = cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    continue

                # Publie la frame fraiche pour YOLO + snapshot des detections, sous verrou.
                with lock:
                    latest_frame = frame
                    latest_frame_id += 1
                    current_detections = list(detections)

                # Les boxes sont ceux de la derniere inference, mais l'image est toujours recente.
                # On dessine sur une copie : latest_frame reste intacte pour la prochaine inference YOLO.
                web_frame = frame.copy()
                draw_detections(web_frame, current_detections)
                # FPS en haut a droite : le chip HTML "REC" occupe le coin haut-gauche et masquait l'ancien texte.
                fps_text = f'{fps_display:.0f} FPS - {len(current_detections)} obj'
                (fps_tw, fps_th), fps_bl = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                fps_x, fps_y = web_frame.shape[1] - fps_tw - 18, 10  # marge de 10 px depuis le bord droit/haut
                cv2.rectangle(web_frame, (fps_x - 8, fps_y), (fps_x + fps_tw + 8, fps_y + fps_th + fps_bl + 10), (10, 12, 16), -1)  # fond sombre, comme les etiquettes YOLO
                cv2.putText(web_frame, fps_text, (fps_x, fps_y + fps_th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2, cv2.LINE_AA)
                # Re-encode pour le web (qualite 60) : reglage d'affichage, independant du JPEG reseau.
                ret, buffer = cv2.imencode(
                    '.jpg', web_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if not ret:
                    continue

                # Publie la frame annotee pour le generateur MJPEG.
                with lock:
                    output_frame = buffer.tobytes()
                    output_frame_id += 1
                # MAJ FPS : fige la valeur a chaque fenetre d'1 s ecoulee, puis repart.
                fps_count += 1
                now = time.monotonic()
                if now - fps_t0 >= 1.0:
                    fps_display = fps_count / (now - fps_t0)
                    fps_count, fps_t0, fps_last_update = 0, now, now
            # Paquet corrompu / decode rate : on jette juste cette frame, le flux continue.
            except Exception as error:
                app.logger.warning('Frame UDP ignoree: %s', error)
    # Erreur socket (port occupe, reseau coupe) : log + sortie, le finally ferme le socket.
    except OSError as error:
        app.logger.exception('Reception UDP arretee: %s', error)
    finally:
        udp_socket.close()


def run_detection(model):
    """Execute YOLO sur le dernier frame disponible, sans mettre UDP en attente."""
    global detections, nb_objets
    processed_frame_id = 0

    # Tourne a son rythme : traite uniquement la frame la plus recente, jamais la file entiere.
    while True:
        with lock:
            # Anti-retraitement : pas de nouvelle frame depuis le dernier passage -> on attend.
            if latest_frame is None or latest_frame_id == processed_frame_id:
                frame_to_process = None
            else:
                # Copie sous verrou puis inference hors verrou : UDP et Flask ne sont jamais bloques.
                frame_to_process = latest_frame.copy()
                frame_id = latest_frame_id

        if frame_to_process is None:
            time.sleep(0.01)
            continue

        try:
            # imgsz=320 : compromis vitesse/precision (coherent avec le warmup du __main__) ; [0] = 1ere image du batch.
            result = model(frame_to_process, imgsz=320, verbose=False)[0]
            current_detections = []
            for box in result.boxes:
                # Coordonnees pixels (haut-gauche -> bas-droit), rapatriees sur CPU puis en int pour OpenCV.
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].item())
                # Stocke (box, label texte via result.names, score) : format consomme par draw_detections.
                current_detections.append((
                    x1, y1, x2, y2,
                    result.names[class_id],
                    float(box.conf[0].item()),
                ))

            # Remplace atomiquement la liste : receive_frames lit toujours une liste coherente.
            with lock:
                detections = current_detections
                nb_objets = len(current_detections)
        except Exception as error:
            # Une frame invalide ne doit pas arreter le service video.
            app.logger.exception('Erreur YOLO: %s', error)
        finally:
            # Marque la frame comme traitee meme en cas d'erreur : evite de boucler dessus.
            processed_frame_id = frame_id


def generate_web_stream():
    """Générateur de flux HTTP MJPEG pour les clients web"""
    # Retient la derniere frame envoyee pour ne streamer que les nouvelles (pas de doublons).
    last_sent_id = 0
    while True:
        with lock:
            frame_bytes = output_frame
            frame_id = output_frame_id

        # Pas encore de frame, ou deja envoyee : on attend sans spammer le client.
        if frame_bytes is None or frame_id == last_sent_id:
            time.sleep(0.01)
            continue

        last_sent_id = frame_id

        # Envoie l'image au format multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# Page HTML : un simple <img> qui consomme le flux MJPEG ci-dessous (+ overlays/panneaux statiques).
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
    <title>Relais YOLO - Live</title>
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
            width:100%; /* pleine largeur de la carte : plus de fond gris sur les cotes */
            aspect-ratio: 4 / 3; /* repli avant la 1re frame, corrige en JS au ratio reel du flux */
            overflow:hidden;
        }
        .video-wrap img{
            position:absolute; inset:0;
            width:100%; height:100%;
            object-fit:contain; /* remplit exactement des que le ratio = ratio flux ; meme fond sinon */
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
                <h2>Flux <span>en direct</span> - réseau local</h2>
                <p>Interface responsive : le flux s'adapte à la taille de l'écran sans déformation.</p>
            </div>
            <div class="stats">
                <span class="stat">UDP <b>:5000</b></span>
                <span class="stat">HTTP <b>:8000</b></span>
                <span class="stat">YOLO <b>v8n</b></span>
                <span class="stat">FPS <b id="fps">–</b></span>
                <span class="stat">Objets <b id="objets">0</b></span>
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
                <div>Flux MJPEG en temps réel &middot; <strong>ratio natif caméra</strong> &middot;</div>
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
        const wrap = document.getElementById('wrap');
        let okOnce = false;
        img.addEventListener('load', () => {
            if(!okOnce){ ph.hidden = true; okOnce = true; }
            if(img.naturalWidth){ res.textContent = img.naturalWidth + ' × ' + img.naturalHeight + ' · MJPEG'; wrap.style.aspectRatio = img.naturalWidth + ' / ' + img.naturalHeight; }
        });
        img.addEventListener('error', () => { ph.hidden = false; });
        // Badge FPS : interroge /fps chaque seconde (requete negligeable).
        const fpsEl = document.getElementById('fps');
        const objetsEl = document.getElementById('objets');
        let retryAt = 0; // Prochaine reconnexion autorisee (anti-spam quand le serveur est down).
        setInterval(async () => {
            try {
                const d = (await (await fetch('/fps', {cache: 'no-store'})).json());
                fpsEl.textContent = d.fps; objetsEl.textContent = d.objets;
            } catch (e) { /* /fps injoignable = recepteur coupe : relance le flux, throttle 5 s */
                if (Date.now() > retryAt) { retryAt = Date.now() + 5000; img.src = '/video_feed?t=' + Date.now(); }
            }
        }, 1000);
        // Auto-retry discret si le flux coupe (cache-bust)
        setInterval(() => {
            if(img.naturalWidth === 0 && okOnce){
                const u = new URL(img.src, location.href);
                u.searchParams.set('t', Date.now());
                img.src = u; // Rejoue la requete : sans cette ligne le retry ne faisait rien.
            }
        }, 5000);
    </script>
</body>
</html>
    ''')


# Endpoint MJPEG : chaque `yield` envoie une image JPEG delimitee par `--frame`.
@app.route('/video_feed')
def video_feed():
    """Endpoint HTTP diffusant le flux vidéo"""
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# FPS courant : 0 si le flux est coupe depuis plus de 2 s (evite une valeur fige).
@app.route('/fps')
def fps():
    """FPS + objets detectes (0 si flux coupe depuis plus de 2 s)."""
    frais = time.monotonic() - fps_last_update <= 2.0
    return {'fps': round(fps_display, 1) if frais else 0.0, 'objets': nb_objets if frais else 0}


# Point d'entree : warmup YOLO + 2 threads daemon + serveur Flask (appel bloquant).
if __name__ == '__main__':
    print("Chargement et préparation du modèle YOLO...")
    # Charge le modele nano (leger, temps reel) depuis le poids local versionne dans le repo.
    model_global = YOLO('yolov8n.pt')
    
    # Inférence à vide (dummy frame) pour forcer PyTorch à allouer la mémoire
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    model_global(dummy_frame, imgsz=320, verbose=False)
    print("Modèle YOLO prêt ! Lancement des services...")
    # La reception et l'inference sont separees pour preserver une faible latence.
    threading.Thread(target=receive_frames, daemon=True).start()
    threading.Thread(target=run_detection,args=(model_global,), daemon=True).start()

    # Lancement du serveur Web Flask sur le port 8000
    # Accessible via http://<IP_DE_CE_PC>:8000 sur le réseau local
    app.run(host='0.0.0.0', port=8000, debug=False)
