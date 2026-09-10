import socket
import cv2
import numpy as np
import threading
import time

from collections import deque
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, request, jsonify


# ============================================================
# CONFIGURATION
# ============================================================

app = Flask(__name__)

UDP_HOST = "0.0.0.0"
UDP_PORT = 5000

MAX_DATAGRAM_SIZE = 65535
JPEG_QUALITY = 70

MODEL_NAME = "yolo11n.pt"

# Taille envoyée à YOLO.
# Diminuer cette valeur augmente fortement les FPS,
# mais peut diminuer la précision.
YOLO_IMAGE_SIZE = 320

YOLO_CONFIDENCE = 0.6


# ============================================================
# ÉTAT GLOBAL
# ============================================================

lock = threading.Lock()
stop_event = threading.Event()

# Dernière frame UDP reçue
latest_frame = None
latest_frame_id = 0

# Frame JPEG finale envoyée à Flask
output_frame = None
output_frame_id = 0

# Détections
detections_original = []
detections_filtered = []

# Mesure FPS de traitement complet
frame_times = deque()


# ============================================================
# FILTRES
# ============================================================

# Chaque filtre possède :
# - min
# - max
# - step
# - valeur initiale
#
# On garde une intensité indépendante pour chaque filtre.

FILTERS = [
    {
        "name": "Niveaux de gris",
        "short": "Gris",
        "min": 0,
        "max": 100,
        "step": 10,
        "value": 100,
        "unit": "%",
    },
    {
        "name": "Flou gaussien",
        "short": "Flou",
        "min": 1,
        "max": 31,
        "step": 2,
        "value": 5,
        "unit": "px",
    },
    {
        "name": "Détection de contours",
        "short": "Contours",
        "min": 10,
        "max": 250,
        "step": 10,
        "value": 100,
        "unit": "",
    },
    {
        "name": "Seuillage",
        "short": "Seuillage",
        "min": 0,
        "max": 255,
        "step": 10,
        "value": 128,
        "unit": "",
    },
    {
        "name": "Bruit aléatoire",
        "short": "Bruit",
        "min": 0,
        "max": 100,
        "step": 5,
        "value": 20,
        "unit": "σ",
    },
    {
        "name": "Luminosité",
        "short": "Luminosite",
        "min": -100,
        "max": 100,
        "step": 10,
        "value": 0,
        "unit": "",
    },
    {
        "name": "Contraste",
        "short": "Contraste",
        "min": 20,
        "max": 200,
        "step": 10,
        "value": 100,
        "unit": "%",
    },
    {
        "name": "Pixellisation",
        "short": "Pixel",
        "min": 1,
        "max": 30,
        "step": 1,
        "value": 5,
        "unit": "x",
    },
    {
        "name": "Canal rouge",
        "short": "Rouge",
        "min": 0,
        "max": 100,
        "step": 10,
        "value": 50,
        "unit": "%",
    },
    {
        "name": "Compression JPEG",
        "short": "JPEG",
        "min": 0,
        "max": 95,
        "step": 5,
        "value": 20,
        "unit": "%",
    },
]

current_filter_index = 0


# ============================================================
# FILTRAGE
# ============================================================


def apply_filter(frame, filter_index, intensity):
    """
    Applique le filtre sélectionné à l'image.

    Paramètres
    ----------
    frame : np.ndarray
        Image originale au format BGR.

    filter_index : int
        Index du filtre dans FILTERS.

    intensity : int | float
        Intensité actuelle du filtre.

    Retour
    ------
    np.ndarray
        Image filtrée au format BGR (3 canaux).
    """

    filter_name = FILTERS[filter_index]["short"]

    # ========================================================
    # 1 — NIVEAUX DE GRIS PROGRESSIFS
    # ========================================================
    #
    # 0 %   -> image originale
    # 100 % -> complètement en niveaux de gris
    #
    if filter_name == "Gris":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        alpha = np.clip(intensity / 100.0, 0.0, 1.0)

        result = cv2.addWeighted(frame, 1.0 - alpha, gray_bgr, alpha, 0)

        return result

    # ========================================================
    # 2 — FLOU GAUSSIEN
    # ========================================================
    #
    # 1  -> pratiquement aucun flou
    # 5  -> léger
    # 21 -> important
    # 51 -> très important
    #
    elif filter_name == "Flou":
        kernel = int(intensity)

        # GaussianBlur exige un kernel > 0
        kernel = max(1, kernel)

        # Le kernel doit être impair :
        # 1, 3, 5, 7, ...
        if kernel % 2 == 0:
            kernel += 1

        result = cv2.GaussianBlur(frame, (kernel, kernel), 0)

        return result

    # ========================================================
    # 3 — DÉTECTION DE CONTOURS
    # ========================================================
    #
    # Canny supprime presque toute l'information de couleur
    # et conserve principalement les contours.
    #
    # Une intensité plus élevée augmente le seuil nécessaire
    # pour considérer quelque chose comme un contour.
    #
    elif filter_name == "Contours":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        threshold_low = int(np.clip(intensity, 1, 255))

        threshold_high = min(255, threshold_low * 2)

        edges = cv2.Canny(gray, threshold_low, threshold_high)

        # Canny retourne une image à un seul canal.
        # On la reconvertit vers BGR pour YOLO + hstack.
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return result

    # ========================================================
    # 4 — SEUILLAGE
    # ========================================================
    #
    # Tous les pixels sous le seuil deviennent noirs.
    # Tous les pixels au-dessus deviennent blancs.
    #
    elif filter_name == "Seuillage":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        threshold_value = int(np.clip(intensity, 0, 255))

        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        return result

    # ========================================================
    # 5 — BRUIT ALÉATOIRE
    # ========================================================
    #
    # Filtre personnel.
    #
    # intensity correspond à sigma du bruit gaussien.
    #
    # 0   -> aucun bruit
    # 20  -> léger
    # 50  -> important
    # 100 -> très important
    #
    elif filter_name == "Bruit":
        sigma = max(0.0, float(intensity))

        noise = np.random.normal(loc=0, scale=sigma, size=frame.shape).astype(
            np.float32
        )

        noisy = frame.astype(np.float32) + noise

        result = np.clip(noisy, 0, 255).astype(np.uint8)

        return result

    # ========================================================
    # 6 — ASSOMBRISSEMENT
    # ========================================================
    #
    # 0 %   -> luminosité normale
    # 50 %  -> deux fois plus sombre
    # 100 % -> image noire
    #
    elif filter_name == "Luminosite":
        degradation = np.clip(intensity / 100.0, 0.0, 1.0)

        brightness_factor = 1.0 - degradation

        result = frame.astype(np.float32) * brightness_factor

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    # ========================================================
    # 7 — RÉDUCTION DU CONTRASTE
    # ========================================================
    #
    # 0 %   -> contraste original
    # 50 %  -> contraste fortement réduit
    # 100 % -> image presque uniforme
    #
    # Le contraste est réduit autour de la valeur 128.
    #
    elif filter_name == "Contraste":
        degradation = np.clip(intensity / 100.0, 0.0, 1.0)

        contrast_factor = 1.0 - degradation

        image = frame.astype(np.float32)

        result = 128.0 + contrast_factor * (image - 128.0)

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    # ========================================================
    # 8 — PIXELLISATION
    # ========================================================
    #
    # On réduit fortement la résolution,
    # puis on agrandit l'image avec INTER_NEAREST.
    #
    # intensity :
    #
    # 1  -> pratiquement original
    # 5  -> légère pixellisation
    # 15 -> forte
    # 40 -> très forte
    #
    elif filter_name == "Pixel":
        factor = max(1, int(intensity))

        height, width = frame.shape[:2]

        small_width = max(1, width // factor)

        small_height = max(1, height // factor)

        # Réduction
        small = cv2.resize(
            frame, (small_width, small_height), interpolation=cv2.INTER_AREA
        )

        # Agrandissement sans interpolation douce
        # pour conserver les gros pixels.
        result = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

        return result

    # ========================================================
    # 9 — CANAL ROUGE
    # ========================================================
    #
    # OpenCV utilise BGR :
    #
    # channel 0 = Bleu
    # channel 1 = Vert
    # channel 2 = Rouge
    #
    # 0 %   -> image originale
    # 100 % -> uniquement le canal rouge
    #
    elif filter_name == "Rouge":
        strength = np.clip(intensity / 100.0, 0.0, 1.0)

        result = frame.copy().astype(np.float32)

        # On diminue progressivement le bleu
        # et le vert.
        result[:, :, 0] *= 1.0 - strength

        result[:, :, 1] *= 1.0 - strength

        # Canal rouge result[:, :, 2]
        # reste inchangé.

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    # ========================================================
    # 10 — COMPRESSION JPEG
    # ========================================================
    #
    # intensity représente ici la DÉGRADATION.
    #
    # intensity 0  -> JPEG quality 100
    # intensity 20 -> JPEG quality 80
    # intensity 50 -> JPEG quality 50
    # intensity 95 -> JPEG quality 5
    #
    elif filter_name == "JPEG":
        degradation = int(np.clip(intensity, 0, 95))

        quality = max(5, 100 - degradation)

        success, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )

        # En cas d'échec de compression,
        # on retourne simplement l'image originale.
        if not success:
            return frame.copy()

        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if result is None:
            return frame.copy()

        return result

    # ========================================================
    # FILTRE INCONNU
    # ========================================================

    return frame.copy()


# ============================================================
# YOLO
# ============================================================


def extract_detections(result):
    """
    Transforme le résultat Ultralytics en liste simple :
    (x1, y1, x2, y2, label, confidence)
    """

    result_detections = []

    if result.boxes is None:
        return result_detections

    for box in result.boxes:
        coordinates = box.xyxy[0].cpu().numpy().astype(int)

        x1, y1, x2, y2 = coordinates

        class_id = int(box.cls[0].item())

        confidence = float(box.conf[0].item())

        label = result.names[class_id]

        result_detections.append((x1, y1, x2, y2, label, confidence))

    return result_detections


def draw_detections(frame, current_detections):
    """
    Dessine les bounding boxes et les labels YOLO.
    """

    for x1, y1, x2, y2, label, confidence in current_detections:
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 136), 2)

        text = f"{label} {confidence:.0%}"

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        label_top = max(y1, text_height + baseline + 8)

        # Fond du label
        cv2.rectangle(
            frame,
            (x1, label_top - text_height - baseline - 8),
            (x1 + text_width + 10, label_top),
            (0, 255, 136),
            -1,
        )

        # Texte
        cv2.putText(
            frame,
            text,
            (x1 + 5, label_top - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 12, 16),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# OVERLAY
# ============================================================


def draw_panel_title(frame, title, subtitle=""):
    """
    Ajoute un titre en haut d'une image.
    """

    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        title,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if subtitle:
        cv2.putText(
            frame,
            subtitle,
            (18, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 205, 215),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# UDP RECEIVER
# ============================================================


def receive_frames():
    """
    Reçoit les images envoyées par emeteur.py.

    Le socket est vidé à chaque boucle afin de garder uniquement
    la frame la plus récente et éviter d'accumuler du retard.
    """

    global latest_frame
    global latest_frame_id

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)

    udp_socket.bind((UDP_HOST, UDP_PORT))

    udp_socket.setblocking(False)

    print(f"UDP : écoute sur {UDP_HOST}:{UDP_PORT}")

    try:
        while not stop_event.is_set():
            packet = None

            # Vide le buffer UDP et conserve
            # uniquement le paquet le plus récent.
            while True:
                try:
                    data, _ = udp_socket.recvfrom(MAX_DATAGRAM_SIZE)

                    packet = data

                except BlockingIOError:
                    break

            if packet is None:
                time.sleep(0.002)
                continue

            try:
                frame = cv2.imdecode(
                    np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR
                )

                if frame is None:
                    continue

                if frame.size == 0:
                    continue

                with lock:
                    latest_frame = frame

                    latest_frame_id += 1

            except Exception as error:
                app.logger.warning("Frame UDP ignorée : %s", error)

    except OSError as error:
        if not stop_event.is_set():
            app.logger.exception("Erreur UDP : %s", error)

    finally:
        udp_socket.close()

        print("Socket UDP fermé.")


# ============================================================
# THREAD DE DÉTECTION
# ============================================================


def run_detection(model):
    """
    Pipeline principal :

        frame
          |
          +---- ORIGINAL ----+
          |                  |
          +--> FILTRE -------+
                             |
                       YOLO batch
                             |
                +------------+------------+
                |                         |
             original                  filtered
                |                         |
              boxes                     boxes
                |                         |
                +---------- hstack -------+
                             |
                           MJPEG
    """

    global output_frame
    global output_frame_id

    global detections_original
    global detections_filtered

    processed_frame_id = -1

    while not stop_event.is_set():
        # ----------------------------------------------------
        # Récupération de la dernière frame
        # ----------------------------------------------------

        with lock:
            if latest_frame is None or latest_frame_id == processed_frame_id:
                frame_to_process = None

            else:
                frame_to_process = latest_frame.copy()

                frame_id = latest_frame_id

                filter_index = current_filter_index

                intensity = FILTERS[filter_index]["value"]

        if frame_to_process is None:
            time.sleep(0.002)

            continue

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        original = frame_to_process

        # ----------------------------------------------------
        # FILTRÉ
        # ----------------------------------------------------

        filtered = apply_filter(original, filter_index, intensity)

        try:
            # =================================================
            # UNE SEULE COPIE DU MODÈLE
            #
            # Deux images sont envoyées au modèle :
            #   index 0 -> original
            #   index 1 -> filtered
            #
            # Cela réalise deux détections tout en permettant
            # au framework d'utiliser un batch.
            # =================================================

            results = model(
                [original, filtered],
                imgsz=YOLO_IMAGE_SIZE,
                conf=YOLO_CONFIDENCE,
                verbose=False,
            )

            result_original = results[0]
            result_filtered = results[1]

            current_original_detections = extract_detections(result_original)

            current_filtered_detections = extract_detections(result_filtered)

            # ------------------------------------------------
            # Dessin
            # ------------------------------------------------

            left = original.copy()

            right = filtered.copy()

            draw_detections(left, current_original_detections)

            draw_detections(right, current_filtered_detections)

            filter_config = FILTERS[filter_index]

            intensity_text = f"{intensity}{filter_config['unit']}"

            draw_panel_title(
                left, "ORIGINAL", (f"{len(current_original_detections)} objet(s)")
            )

            draw_panel_title(
                right,
                filter_config["name"].upper(),
                (
                    f"Intensité : {intensity_text} | "
                    f"{len(current_filtered_detections)} "
                    f"objet(s)"
                ),
            )

            # Ligne séparatrice
            separator = np.full((left.shape[0], 4, 3), 40, dtype=np.uint8)

            combined = np.hstack((left, separator, right))

            # ------------------------------------------------
            # Compression pour Flask
            # ------------------------------------------------

            success, buffer = cv2.imencode(
                ".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            if not success:
                continue

            now = time.monotonic()

            with lock:
                detections_original = current_original_detections

                detections_filtered = current_filtered_detections

                output_frame = buffer.tobytes()

                output_frame_id += 1

                frame_times.append(now)

                # FPS lissé sur les deux
                # dernières secondes.
                while frame_times and now - frame_times[0] > 2:
                    frame_times.popleft()

        except Exception as error:
            app.logger.exception("Erreur YOLO : %s", error)

        finally:
            processed_frame_id = frame_id


# ============================================================
# MJPEG
# ============================================================


def generate_web_stream():
    """
    Générateur MJPEG pour Flask.
    """

    last_sent_id = -1

    while not stop_event.is_set():
        with lock:
            frame_bytes = output_frame

            frame_id = output_frame_id

        if frame_bytes is None or frame_id == last_sent_id:
            time.sleep(0.005)

            continue

        last_sent_id = frame_id

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# ============================================================
# FLASK — INTERFACE
# ============================================================


@app.route("/")
def index():
    return render_template_string(
        """
 <!DOCTYPE html>
<html lang="fr">

<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <meta name="color-scheme" content="dark">

    <title>IA03 — Robustesse YOLO</title>

    <style>

        /* =========================================================
           VARIABLES
        ========================================================= */

        :root {
            --bg: #080a0f;
            --bg-soft: #10131a;

            --card: rgba(19, 23, 31, 0.94);
            --card-light: rgba(27, 32, 43, 0.94);

            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.17);

            --text: #f5f7fb;
            --muted: #8f98a8;

            --green: #00e68a;
            --green-soft: rgba(0, 230, 138, 0.12);

            --blue: #38bdf8;
            --blue-soft: rgba(56, 189, 248, 0.12);

            --orange: #f59e0b;
            --orange-soft: rgba(245, 158, 11, 0.12);

            --red: #ff5c70;
            --red-soft: rgba(255, 92, 112, 0.12);

            --purple: #a78bfa;

            --radius: 18px;

            --shadow:
                0 20px 65px rgba(0, 0, 0, 0.38);
        }


        /* =========================================================
           RESET
        ========================================================= */

        * {
            box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            margin: 0;

            min-height: 100vh;

            background:
                radial-gradient(
                    850px 550px at 5% -10%,
                    rgba(0, 230, 138, 0.11),
                    transparent 60%
                ),
                radial-gradient(
                    900px 600px at 100% 0%,
                    rgba(56, 189, 248, 0.09),
                    transparent 60%
                ),
                var(--bg);

            color: var(--text);

            font-family:
                Inter,
                ui-sans-serif,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            -webkit-font-smoothing: antialiased;
        }


        button {
            font: inherit;
        }


        /* =========================================================
           HEADER
        ========================================================= */

        header {
            position: sticky;

            top: 0;

            z-index: 100;

            background:
                rgba(8, 10, 15, 0.78);

            border-bottom:
                1px solid var(--border);

            backdrop-filter:
                blur(18px)
                saturate(1.4);
        }


        .header-inner {
            max-width: 1550px;

            margin: 0 auto;

            padding:
                15px
                24px;

            display: flex;

            align-items: center;

            justify-content: space-between;

            gap: 20px;
        }


        .brand {
            display: flex;

            align-items: center;

            gap: 13px;
        }


        .logo {
            width: 43px;

            height: 43px;

            display: grid;

            place-items: center;

            border-radius: 12px;

            color: #06110d;

            background:
                linear-gradient(
                    135deg,
                    var(--green),
                    var(--blue)
                );

            font-weight: 900;

            box-shadow:
                0 8px 30px
                rgba(0, 230, 138, 0.2);
        }


        .brand h1 {
            margin: 0;

            font-size: 18px;

            font-weight: 760;

            letter-spacing: -0.02em;
        }


        .brand p {
            margin:
                3px 0 0;

            color: var(--muted);

            font-size: 12px;
        }


        .connection {
            display: flex;

            align-items: center;

            gap: 9px;

            padding:
                8px 13px;

            border:
                1px solid
                rgba(0, 230, 138, 0.25);

            border-radius: 999px;

            background:
                var(--green-soft);

            color: #a8f5d0;

            font-size: 12px;

            font-weight: 650;
        }


        .connection.error {
            color: #ffb8c1;

            border-color:
                rgba(255, 92, 112, 0.25);

            background:
                var(--red-soft);
        }


        .connection-dot {
            width: 8px;

            height: 8px;

            border-radius: 50%;

            background: var(--green);

            animation:
                pulse
                1.8s
                infinite;
        }


        .connection.error .connection-dot {
            background: var(--red);

            animation: none;
        }


        @keyframes pulse {
            0% {
                box-shadow:
                    0 0 0 0
                    rgba(0, 230, 138, 0.4);
            }

            70% {
                box-shadow:
                    0 0 0 9px
                    rgba(0, 230, 138, 0);
            }

            100% {
                box-shadow:
                    0 0 0 0
                    rgba(0, 230, 138, 0);
            }
        }


        /* =========================================================
           MAIN
        ========================================================= */

        main {
            width: 100%;

            max-width: 1550px;

            margin: 0 auto;

            padding: 24px;
        }


        .top-row {
            display: flex;

            justify-content: space-between;

            align-items: flex-end;

            gap: 20px;

            margin-bottom: 18px;
        }


        .page-title h2 {
            margin: 0;

            font-size:
                clamp(
                    25px,
                    3vw,
                    36px
                );

            line-height: 1.1;

            letter-spacing: -0.04em;
        }


        .page-title h2 span {
            background:
                linear-gradient(
                    90deg,
                    var(--green),
                    var(--blue)
                );

            -webkit-background-clip: text;

            -webkit-text-fill-color: transparent;
        }


        .page-title p {
            margin:
                8px 0 0;

            color: var(--muted);

            font-size: 14px;
        }


        /* =========================================================
           STATS
        ========================================================= */

        .stats-grid {
            display: grid;

            grid-template-columns:
                1.35fr
                1fr
                1fr
                1fr
                0.8fr;

            gap: 12px;

            margin-bottom: 18px;
        }


        .stat-card {
            min-height: 96px;

            padding: 16px;

            border:
                1px solid var(--border);

            border-radius: var(--radius);

            background:
                linear-gradient(
                    180deg,
                    var(--card-light),
                    var(--card)
                );
        }


        .stat-label {
            color: var(--muted);

            font-size: 10px;

            font-weight: 750;

            letter-spacing: 0.09em;

            text-transform: uppercase;
        }


        .stat-value {
            margin-top: 9px;

            overflow: hidden;

            font-size: 21px;

            font-weight: 820;

            letter-spacing: -0.03em;

            text-overflow: ellipsis;

            white-space: nowrap;
        }


        .stat-value.green {
            color: var(--green);
        }


        .stat-value.blue {
            color: var(--blue);
        }


        .stat-value.orange {
            color: var(--orange);
        }


        .stat-sub {
            margin-top: 4px;

            color: var(--muted);

            font-size: 11px;
        }


        /* =========================================================
           VIDEO
        ========================================================= */

        .video-card {
            overflow: hidden;

            border:
                1px solid var(--border);

            border-radius: 20px;

            background: black;

            box-shadow: var(--shadow);
        }


        .video-header {
            min-height: 49px;

            padding:
                11px
                15px;

            display: flex;

            align-items: center;

            justify-content: space-between;

            gap: 10px;

            background:
                rgba(17, 20, 27, 0.98);

            border-bottom:
                1px solid var(--border);
        }


        .video-label {
            display: flex;

            align-items: center;

            gap: 9px;

            color: #d2d8e1;

            font-size: 12px;

            font-weight: 650;
        }


        .live {
            padding:
                5px
                8px;

            border-radius: 7px;

            background:
                var(--red-soft);

            color: #ff8795;

            font-size: 10px;

            font-weight: 850;

            letter-spacing: 0.1em;
        }


        .model-badge {
            color: var(--muted);

            font-size: 11px;
        }


        .video-container {
            position: relative;

            background: #000;
        }


        .video-container img {
            display: block;

            width: 100%;

            object-fit: contain;

            background: #000;
        }


        /* =========================================================
           CONTROLS GRID
        ========================================================= */

        .bottom-grid {
            display: grid;

            grid-template-columns:
                minmax(0, 3fr)
                minmax(280px, 1fr);

            gap: 18px;

            margin-top: 18px;
        }


        .panel {
            padding: 19px;

            border:
                1px solid var(--border);

            border-radius: var(--radius);

            background:
                linear-gradient(
                    180deg,
                    var(--card-light),
                    var(--card)
                );
        }


        .panel-title {
            display: flex;

            align-items: center;

            justify-content: space-between;

            gap: 15px;

            margin-bottom: 15px;
        }


        .panel-title h3 {
            margin: 0;

            font-size: 15px;
        }


        .panel-title span {
            color: var(--muted);

            font-size: 11px;
        }


        /* =========================================================
           FILTER GRID
        ========================================================= */

        .filter-grid {
            display: grid;

            grid-template-columns:
                repeat(5, minmax(0, 1fr));

            gap: 9px;
        }


        .filter-button {
            position: relative;

            min-height: 86px;

            padding:
                12px
                11px;

            text-align: left;

            border:
                1px solid var(--border);

            border-radius: 12px;

            color: #cbd2dd;

            background:
                rgba(255, 255, 255, 0.032);

            cursor: pointer;

            transition:
                transform 0.15s ease,
                border-color 0.15s ease,
                background 0.15s ease;
        }


        .filter-button:hover {
            transform:
                translateY(-2px);

            border-color:
                var(--border-hover);

            background:
                rgba(255, 255, 255, 0.065);
        }


        .filter-button.active {
            color: #d7ffec;

            border-color:
                rgba(0, 230, 138, 0.42);

            background:
                linear-gradient(
                    145deg,
                    rgba(0, 230, 138, 0.15),
                    rgba(56, 189, 248, 0.05)
                );

            box-shadow:
                inset 0 0 0 1px
                rgba(0, 230, 138, 0.05);
        }


        .filter-top {
            display: flex;

            align-items: center;

            justify-content: space-between;

            gap: 8px;

            margin-bottom: 10px;
        }


        .filter-number {
            display: inline-grid;

            width: 24px;

            height: 24px;

            place-items: center;

            border-radius: 7px;

            background:
                rgba(255, 255, 255, 0.075);

            font-size: 10px;

            font-weight: 850;
        }


        .filter-icon {
            color: var(--muted);

            font-size: 17px;
        }


        .filter-name {
            display: block;

            font-size: 12px;

            font-weight: 690;

            line-height: 1.25;
        }


        .filter-description {
            display: block;

            margin-top: 4px;

            color: var(--muted);

            font-size: 10px;

            line-height: 1.3;
        }


        /* =========================================================
           INTENSITY PANEL
        ========================================================= */

        .intensity-panel {
            display: flex;

            flex-direction: column;
        }


        .intensity-display {
            padding: 16px;

            border:
                1px solid var(--border);

            border-radius: 13px;

            background:
                rgba(255, 255, 255, 0.035);
        }


        .intensity-display small {
            display: block;

            margin-bottom: 5px;

            color: var(--muted);

            font-size: 10px;

            font-weight: 700;

            letter-spacing: 0.08em;

            text-transform: uppercase;
        }


        .intensity-big {
            color: var(--green);

            font-size: 30px;

            font-weight: 850;

            letter-spacing: -0.04em;
        }


        .intensity-buttons {
            display: grid;

            grid-template-columns:
                1fr 1fr;

            gap: 9px;

            margin-top: 10px;
        }


        .intensity-button {
            min-height: 55px;

            border:
                1px solid var(--border);

            border-radius: 12px;

            background:
                rgba(255, 255, 255, 0.05);

            color: white;

            cursor: pointer;

            font-size: 23px;

            font-weight: 650;

            transition:
                background 0.15s ease,
                transform 0.1s ease;
        }


        .intensity-button:hover {
            background:
                rgba(255, 255, 255, 0.1);
        }


        .intensity-button:active {
            transform:
                scale(0.97);
        }


        /* =========================================================
           COMPARISON
        ========================================================= */

        .comparison {
            margin-top: 11px;

            padding: 14px;

            border:
                1px solid var(--border);

            border-radius: 12px;

            background:
                rgba(255, 255, 255, 0.025);
        }


        .comparison-label {
            color: var(--muted);

            font-size: 10px;

            font-weight: 700;

            letter-spacing: 0.08em;

            text-transform: uppercase;
        }


        .comparison-result {
            margin-top: 6px;

            font-size: 14px;

            font-weight: 750;
        }


        .comparison-result.good {
            color: var(--green);
        }


        .comparison-result.warning {
            color: var(--orange);
        }


        .comparison-result.bad {
            color: var(--red);
        }


        .experiment-info {
            margin-top: 11px;

            padding:
                12px 14px;

            border:
                1px solid var(--border);

            border-radius: 12px;

            color: var(--muted);

            background:
                rgba(56, 189, 248, 0.035);

            font-size: 11px;

            line-height: 1.55;
        }


        .experiment-info strong {
            color: #ccd5e0;
        }


        /* =========================================================
           SHORTCUTS
        ========================================================= */

        .shortcuts {
            margin-top: 18px;

            display: flex;

            flex-wrap: wrap;

            justify-content: center;

            gap: 5px;
        }


        .shortcut {
            display: flex;

            align-items: center;

            gap: 5px;

            padding: 5px;

            color: var(--muted);

            font-size: 10px;
        }


        kbd {
            min-width: 26px;

            padding:
                4px
                7px;

            text-align: center;

            border:
                1px solid
                rgba(255, 255, 255, 0.12);

            border-bottom-color:
                rgba(255, 255, 255, 0.22);

            border-radius: 6px;

            background:
                rgba(255, 255, 255, 0.055);

            color: #e8ecf3;

            font-family: inherit;

            font-size: 10px;
        }


        /* =========================================================
           FOOTER
        ========================================================= */

        footer {
            padding:
                28px
                20px;

            color: #626a77;

            text-align: center;

            font-size: 11px;
        }


        /* =========================================================
           RESPONSIVE
        ========================================================= */

        @media (max-width: 1100px) {

            .stats-grid {
                grid-template-columns:
                    repeat(3, 1fr);
            }

            .bottom-grid {
                grid-template-columns: 1fr;
            }

            .filter-grid {
                grid-template-columns:
                    repeat(5, 1fr);
            }

        }


        @media (max-width: 850px) {

            .filter-grid {
                grid-template-columns:
                    repeat(3, 1fr);
            }

        }


        @media (max-width: 650px) {

            main {
                padding: 14px;
            }

            .header-inner {
                padding:
                    12px
                    14px;
            }

            .brand p {
                display: none;
            }

            .stats-grid {
                grid-template-columns:
                    repeat(2, 1fr);
            }

            .filter-grid {
                grid-template-columns:
                    repeat(2, 1fr);
            }

            .top-row {
                align-items: flex-start;

                flex-direction: column;
            }

        }

    </style>
</head>


<body>


<!-- =========================================================
     HEADER
========================================================= -->

<header>

    <div class="header-inner">

        <div class="brand">

            <div class="logo">
                IA
            </div>

            <div>

                <h1>
                    IA03 Vision Lab
                </h1>

                <p>
                    Filtres d'image & robustesse de détection
                </p>

            </div>

        </div>


        <div
            class="connection"
            id="connectionStatus"
        >

            <span class="connection-dot"></span>

            <span>
                Flux connecté
            </span>

        </div>

    </div>

</header>


<!-- =========================================================
     MAIN
========================================================= -->

<main>


    <div class="top-row">

        <div class="page-title">

            <h2>
                Robustesse du modèle
                <span>YOLO</span>
            </h2>

            <p>
                Analyse en temps réel de l'impact des
                dégradations d'image sur la détection d'objets.
            </p>

        </div>

    </div>


    <!-- =====================================================
         STATS
    ====================================================== -->

    <section class="stats-grid">


        <div class="stat-card">

            <div class="stat-label">
                Filtre actuel
            </div>

            <div
                class="stat-value green"
                id="filter"
            >
                —
            </div>

            <div class="stat-sub">
                Sélection clavier 0–9
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-label">
                Intensité
            </div>

            <div
                class="stat-value blue"
                id="intensity"
            >
                —
            </div>

            <div class="stat-sub">
                ↑ / ↓ pour modifier
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-label">
                Image originale
            </div>

            <div
                class="stat-value"
                id="originalObjects"
            >
                0
            </div>

            <div class="stat-sub">
                objets détectés
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-label">
                Image filtrée
            </div>

            <div
                class="stat-value orange"
                id="filteredObjects"
            >
                0
            </div>

            <div class="stat-sub">
                objets détectés
            </div>

        </div>


        <div class="stat-card">

            <div class="stat-label">
                Performance
            </div>

            <div
                class="stat-value"
                id="fps"
            >
                0
            </div>

            <div class="stat-sub">
                images / seconde
            </div>

        </div>


    </section>


    <!-- =====================================================
         VIDEO
    ====================================================== -->

    <section class="video-card">

        <div class="video-header">

            <div class="video-label">

                <span class="live">
                    LIVE
                </span>

                Original vs image filtrée

            </div>


            <div class="model-badge">
                YOLO11n · seuil de confiance 60 %
            </div>

        </div>


        <div class="video-container">

            <img
                id="videoStream"
                src="/video_feed"
                alt="Flux vidéo IA03"
            >

        </div>

    </section>


    <!-- =====================================================
         CONTROLS
    ====================================================== -->

    <section class="bottom-grid">


        <!-- =================================================
             FILTERS
        ================================================== -->

        <div class="panel">

            <div class="panel-title">

                <h3>
                    Filtres d'image
                </h3>

                <span>
                    10 filtres disponibles
                </span>

            </div>


            <div class="filter-grid">


                <!-- 1 -->

                <button
                    class="filter-button"
                    data-filter="0"
                    onclick="setFilter(0)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            1
                        </span>

                        <span class="filter-icon">
                            ◐
                        </span>

                    </div>

                    <span class="filter-name">
                        Niveaux de gris
                    </span>

                    <span class="filter-description">
                        Suppression progressive des couleurs
                    </span>

                </button>


                <!-- 2 -->

                <button
                    class="filter-button"
                    data-filter="1"
                    onclick="setFilter(1)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            2
                        </span>

                        <span class="filter-icon">
                            ◌
                        </span>

                    </div>

                    <span class="filter-name">
                        Flou gaussien
                    </span>

                    <span class="filter-description">
                        Réduction des détails et textures
                    </span>

                </button>


                <!-- 3 -->

                <button
                    class="filter-button"
                    data-filter="2"
                    onclick="setFilter(2)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            3
                        </span>

                        <span class="filter-icon">
                            △
                        </span>

                    </div>

                    <span class="filter-name">
                        Contours
                    </span>

                    <span class="filter-description">
                        Détection des bords avec Canny
                    </span>

                </button>


                <!-- 4 -->

                <button
                    class="filter-button"
                    data-filter="3"
                    onclick="setFilter(3)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            4
                        </span>

                        <span class="filter-icon">
                            ◑
                        </span>

                    </div>

                    <span class="filter-name">
                        Seuillage
                    </span>

                    <span class="filter-description">
                        Conversion en noir et blanc
                    </span>

                </button>


                <!-- 5 -->

                <button
                    class="filter-button"
                    data-filter="4"
                    onclick="setFilter(4)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            5
                        </span>

                        <span class="filter-icon">
                            ⁙
                        </span>

                    </div>

                    <span class="filter-name">
                        Bruit
                    </span>

                    <span class="filter-description">
                        Ajout de bruit gaussien aléatoire
                    </span>

                </button>


                <!-- 6 -->

                <button
                    class="filter-button"
                    data-filter="5"
                    onclick="setFilter(5)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            6
                        </span>

                        <span class="filter-icon">
                            ☾
                        </span>

                    </div>

                    <span class="filter-name">
                        Assombrissement
                    </span>

                    <span class="filter-description">
                        Réduction progressive de luminosité
                    </span>

                </button>


                <!-- 7 -->

                <button
                    class="filter-button"
                    data-filter="6"
                    onclick="setFilter(6)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            7
                        </span>

                        <span class="filter-icon">
                            ◒
                        </span>

                    </div>

                    <span class="filter-name">
                        Contraste
                    </span>

                    <span class="filter-description">
                        Réduction du contraste de l'image
                    </span>

                </button>


                <!-- 8 -->

                <button
                    class="filter-button"
                    data-filter="7"
                    onclick="setFilter(7)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            8
                        </span>

                        <span class="filter-icon">
                            ▦
                        </span>

                    </div>

                    <span class="filter-name">
                        Pixellisation
                    </span>

                    <span class="filter-description">
                        Réduction de la résolution spatiale
                    </span>

                </button>


                <!-- 9 -->

                <button
                    class="filter-button"
                    data-filter="8"
                    onclick="setFilter(8)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            9
                        </span>

                        <span
                            class="filter-icon"
                            style="color:#ff6677"
                        >
                            ●
                        </span>

                    </div>

                    <span class="filter-name">
                        Canal rouge
                    </span>

                    <span class="filter-description">
                        Suppression du bleu et du vert
                    </span>

                </button>


                <!-- 0 -->

                <button
                    class="filter-button"
                    data-filter="9"
                    onclick="setFilter(9)"
                >

                    <div class="filter-top">

                        <span class="filter-number">
                            0
                        </span>

                        <span class="filter-icon">
                            JPG
                        </span>

                    </div>

                    <span class="filter-name">
                        Compression JPEG
                    </span>

                    <span class="filter-description">
                        Ajout d'artefacts de compression
                    </span>

                </button>


            </div>

        </div>


        <!-- =================================================
             INTENSITY
        ================================================== -->

        <div class="panel intensity-panel">

            <div class="panel-title">

                <h3>
                    Intensité
                </h3>

                <span>
                    Réglage en temps réel
                </span>

            </div>


            <div class="intensity-display">

                <small>
                    Valeur actuelle
                </small>

                <div
                    class="intensity-big"
                    id="intensityLarge"
                >
                    —
                </div>

            </div>


            <div class="intensity-buttons">

                <button
                    class="intensity-button"
                    onclick="changeIntensity(-1)"
                    title="Diminuer l'intensité"
                >
                    −
                </button>


                <button
                    class="intensity-button"
                    onclick="changeIntensity(1)"
                    title="Augmenter l'intensité"
                >
                    +
                </button>

            </div>


            <div class="comparison">

                <div class="comparison-label">
                    Impact sur la détection
                </div>

                <div
                    class="comparison-result good"
                    id="comparisonResult"
                >
                    Aucune perte de détection
                </div>

            </div>


            <div class="experiment-info">

                <strong>
                    Objectif :
                </strong>

                augmenter progressivement l'intensité
                jusqu'au moment où le modèle ne détecte
                plus l'objet sur l'image filtrée.

            </div>

        </div>


    </section>


    <!-- =====================================================
         KEYBOARD SHORTCUTS
    ====================================================== -->

    <div class="shortcuts">

        <div class="shortcut">
            <kbd>1</kbd>
            Gris
        </div>

        <div class="shortcut">
            <kbd>2</kbd>
            Flou
        </div>

        <div class="shortcut">
            <kbd>3</kbd>
            Contours
        </div>

        <div class="shortcut">
            <kbd>4</kbd>
            Seuillage
        </div>

        <div class="shortcut">
            <kbd>5</kbd>
            Bruit
        </div>

        <div class="shortcut">
            <kbd>6</kbd>
            Sombre
        </div>

        <div class="shortcut">
            <kbd>7</kbd>
            Contraste
        </div>

        <div class="shortcut">
            <kbd>8</kbd>
            Pixel
        </div>

        <div class="shortcut">
            <kbd>9</kbd>
            Rouge
        </div>

        <div class="shortcut">
            <kbd>0</kbd>
            JPEG
        </div>

        <div class="shortcut">
            <kbd>↑</kbd>
            Intensité +
        </div>

        <div class="shortcut">
            <kbd>↓</kbd>
            Intensité −
        </div>

    </div>


</main>


<footer>
    IA03 · Computer Vision · YOLO11 · Analyse de robustesse en temps réel
</footer>


<script>


    /* =========================================================
       FILTRES
    ========================================================= */

    const filterNames = [
        "Niveaux de gris",
        "Flou gaussien",
        "Détection de contours",
        "Seuillage",
        "Bruit aléatoire",
        "Assombrissement",
        "Réduction du contraste",
        "Pixellisation",
        "Canal rouge",
        "Compression JPEG"
    ];


    /* =========================================================
       CONNECTION
    ========================================================= */

    function setConnectionState(connected) {

        const element =
            document.getElementById(
                "connectionStatus"
            );


        const text =
            element.querySelector(
                "span:last-child"
            );


        if (connected) {

            element.classList.remove(
                "error"
            );

            text.textContent =
                "Flux connecté";

        }

        else {

            element.classList.add(
                "error"
            );

            text.textContent =
                "Connexion perdue";

        }

    }


    /* =========================================================
       ACTIVE FILTER
    ========================================================= */

    function updateActiveFilter(
        filterName
    ) {

        document
            .querySelectorAll(
                ".filter-button"
            )
            .forEach(
                button => {

                    const index =
                        Number(
                            button.dataset.filter
                        );


                    if (
                        filterNames[index]
                        ===
                        filterName
                    ) {

                        button.classList.add(
                            "active"
                        );

                    }

                    else {

                        button.classList.remove(
                            "active"
                        );

                    }

                }
            );

    }


    /* =========================================================
       COMPARISON RESULT
    ========================================================= */

    function updateComparison(
        original,
        filtered
    ) {

        const element =
            document.getElementById(
                "comparisonResult"
            );


        const difference =
            original - filtered;


        element.classList.remove(
            "good",
            "warning",
            "bad"
        );


        if (
            original === 0
            &&
            filtered === 0
        ) {

            element.textContent =
                "Aucun objet détecté";

            element.classList.add(
                "warning"
            );

            return;

        }


        if (difference <= 0) {

            element.textContent =
                "Aucune perte de détection";

            element.classList.add(
                "good"
            );

        }

        else if (difference === 1) {

            element.textContent =
                "−1 objet après filtrage";

            element.classList.add(
                "warning"
            );

        }

        else {

            element.textContent =
                `−${difference} objets après filtrage`;

            element.classList.add(
                "bad"
            );

        }

    }


    /* =========================================================
       UPDATE STATUS
    ========================================================= */

    async function updateStatus() {

        try {

            const response =
                await fetch(
                    "/status",
                    {
                        cache:
                            "no-store"
                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Status unavailable"
                );

            }


            const data =
                await response.json();


            document
                .getElementById(
                    "filter"
                )
                .textContent =
                    data.filter;


            document
                .getElementById(
                    "intensity"
                )
                .textContent =
                    data.intensity;


            document
                .getElementById(
                    "intensityLarge"
                )
                .textContent =
                    data.intensity;


            document
                .getElementById(
                    "originalObjects"
                )
                .textContent =
                    data.original_objects;


            document
                .getElementById(
                    "filteredObjects"
                )
                .textContent =
                    data.filtered_objects;


            document
                .getElementById(
                    "fps"
                )
                .textContent =
                    data.fps;


            updateActiveFilter(
                data.filter
            );


            updateComparison(
                data.original_objects,
                data.filtered_objects
            );


            setConnectionState(
                true
            );

        }

        catch (error) {

            console.error(
                error
            );


            setConnectionState(
                false
            );

        }

    }


    /* =========================================================
       SET FILTER
    ========================================================= */

    async function setFilter(index) {

        try {

            const response =
                await fetch(
                    "/filter",
                    {
                        method:
                            "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                {
                                    index:
                                        index
                                }
                            )
                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Impossible de changer le filtre"
                );

            }


            await updateStatus();

        }

        catch (error) {

            console.error(
                error
            );

        }

    }


    /* =========================================================
       CHANGE INTENSITY
    ========================================================= */

    async function changeIntensity(
        direction
    ) {

        try {

            const response =
                await fetch(
                    "/intensity",
                    {
                        method:
                            "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                {
                                    direction:
                                        direction
                                }
                            )
                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Impossible de modifier l'intensité"
                );

            }


            await updateStatus();

        }

        catch (error) {

            console.error(
                error
            );

        }

    }


    /* =========================================================
       KEYBOARD
    ========================================================= */

    document.addEventListener(
        "keydown",
        function(event) {

            const key =
                event.key;


            /* -------------------------
               Filtres 1 → 9
            -------------------------- */

            if (
                key >= "1"
                &&
                key <= "9"
            ) {

                const index =
                    Number(key) - 1;


                setFilter(
                    index
                );

                return;

            }


            /* -------------------------
               Filtre 10 = touche 0
            -------------------------- */

            if (key === "0") {

                setFilter(
                    9
                );

                return;

            }


            /* -------------------------
               Intensité +
            -------------------------- */

            if (
                key === "+"
                ||
                key === "="
                ||
                key === "ArrowUp"
            ) {

                event.preventDefault();


                changeIntensity(
                    1
                );


                return;

            }


            /* -------------------------
               Intensité -
            -------------------------- */

            if (
                key === "-"
                ||
                key === "_"
                ||
                key === "ArrowDown"
            ) {

                event.preventDefault();


                changeIntensity(
                    -1
                );

            }

        }
    );


    /* =========================================================
       VIDEO CONNECTION
    ========================================================= */

    const video =
        document.getElementById(
            "videoStream"
        );


    video.addEventListener(
        "error",
        function() {

            setConnectionState(
                false
            );

        }
    );


    video.addEventListener(
        "load",
        function() {

            setConnectionState(
                true
            );

        }
    );


    /* =========================================================
       START
    ========================================================= */

    updateStatus();


    setInterval(
        updateStatus,
        500
    );


</script>


</body>
</html>
        """
    )


# ============================================================
# STATUS
# ============================================================


@app.route("/status")
def status():
    with lock:
        now = time.monotonic()

        while frame_times and now - frame_times[0] > 2:
            frame_times.popleft()

        count = len(frame_times)

        if count > 1:
            elapsed = frame_times[-1] - frame_times[0]

            if elapsed > 0:
                fps_value = (count - 1) / elapsed

            else:
                fps_value = 0

        else:
            fps_value = 0

        config = FILTERS[current_filter_index]

        intensity_string = f"{config['value']}{config['unit']}"

        return jsonify(
            {
                "filter": config["name"],
                "intensity": intensity_string,
                "original_objects": len(detections_original),
                "filtered_objects": len(detections_filtered),
                "fps": round(fps_value, 1),
            }
        )


# ============================================================
# API — FILTRE
# ============================================================


@app.route("/filter", methods=["POST"])
def change_filter():
    global current_filter_index

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"status": "error"}), 400

    index = data.get("index")

    if not isinstance(index, int):
        return jsonify({"status": "error"}), 400

    if not (0 <= index < len(FILTERS)):
        return jsonify({"status": "error"}), 400

    with lock:
        current_filter_index = index

        config = FILTERS[current_filter_index]

    print(f"Filtre : {config['name']}")

    return jsonify({"status": "ok", "filter": config["name"]})


# ============================================================
# API — INTENSITÉ
# ============================================================


@app.route("/intensity", methods=["POST"])
def change_intensity():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"status": "error"}), 400

    direction = data.get("direction")

    if direction not in (-1, 1):
        return jsonify({"status": "error"}), 400

    with lock:
        config = FILTERS[current_filter_index]

        new_value = config["value"] + direction * config["step"]

        new_value = max(config["min"], min(config["max"], new_value))

        # Le kernel du GaussianBlur
        # doit rester impair.
        if config["short"] == "Flou" and new_value % 2 == 0:
            if direction > 0:
                new_value += 1

            else:
                new_value -= 1

            new_value = max(config["min"], min(config["max"], new_value))

        config["value"] = new_value

        result_value = f"{config['value']}{config['unit']}"

    return jsonify({"status": "ok", "intensity": result_value})


# ============================================================
# VIDEO FEED
# ============================================================


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_web_stream(), mimetype=("multipart/x-mixed-replace; boundary=frame")
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("IA03 — Robustesse YOLO")
    print("=" * 60)

    print(f"Chargement du modèle {MODEL_NAME}...")

    # --------------------------------------------------------
    # Charge UNE SEULE FOIS le modèle.
    # --------------------------------------------------------

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("Warm-up YOLO...")

    dummy = np.zeros((320, 320, 3), dtype=np.uint8)

    # Batch de deux images,
    # exactement comme pendant l'utilisation.
    model([dummy, dummy], imgsz=YOLO_IMAGE_SIZE, verbose=False)

    print("YOLO prêt.")

    # --------------------------------------------------------
    # Threads
    # --------------------------------------------------------

    udp_thread = threading.Thread(target=receive_frames, daemon=True)

    detection_thread = threading.Thread(
        target=run_detection, args=(model,), daemon=True
    )

    udp_thread.start()

    detection_thread.start()

    print()
    print(f"UDP : port {UDP_PORT}")

    print("Web : http://0.0.0.0:8000")

    print()
    print("Filtres :")

    print("1 = Niveaux de gris")

    print("2 = Flou")

    print("3 = Contours")

    print("4 = Seuillage")

    print("5 = Bruit")

    print()
    print("+ / ↑ = augmenter l'intensité")

    print("- / ↓ = diminuer l'intensité")

    print("=" * 60)
    print()

    try:
        app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)

    finally:
        stop_event.set()

        print("Arrêt du récepteur.")
