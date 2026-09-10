import time

import cv2
import numpy as np
from ultralytics import YOLO

print("1. Chargement du modèle YOLOv11...")
# Au 1er lancement, la ligne suivante télécharge yolo11n.pt (cela peut prendre quelques secondes)
model = YOLO("yolo11n.pt")

print("2. Tentative d'accès à la webcam...")
# Si 0 ne marche pas, essaie avec 1 ou 2
cap = cv2.VideoCapture(0)
# Taille de capture fixée : certaines webcams changent de mode en cours de
# route (ex. 640x480 <-> 1920x1080 selon la lumière), ce qui casse le hstack.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"Capture : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

if not cap.isOpened():
    print("❌ ERREUR : Impossible d'ouvrir la webcam.")
    print("- Vérifie qu'aucune autre application (Teams, Zoom, navigateur) n'utilise la caméra.")
    print("- Essaie d'installer le backend dSHOW sur Windows : cv2.VideoCapture(0, cv2.CAP_DSHOW)")
    print("- Modifie l'index de la caméra : cv2.VideoCapture(1)")
    exit()

print("3. Caméra ouverte !")
print("   - Flèche GAUCHE / DROITE (ou 'a' / 'd') : changer de filtre")
print("   - Flèche HAUT / BAS (ou 'w'/'z' / 's') : intensité + / -")
print("   - 'q' ou Echap : quitter")
print("   (clique d'abord sur la fenêtre vidéo pour lui donner le focus clavier)")

# --- Intensités réglables (HAUT = + d'effet, BAS = - d'effet) ---
blur_k = 15          # noyau impair 1..51, 1 = pas de flou
gray_strength = 1.0  # mélange 0.0 (couleur) .. 1.0 (gris total), pas de 0.1
CANNY_LOW = 30        # seuil fixe ; seuil haut = bas x 2
canny_strength = 1.0  # mélange 0.0 (original) .. 1.0 (contours seuls), pas de 0.1
seuil_strength = 1.0  # mélange 0.0 (original) .. 1.0 (seuil seul), pas de 0.1
shift_px = 20  # décalage max en pixels 0..50, pas de 5 (0 = pas d'effet)

def f_flou(frame):
    k = max(1, blur_k | 1)  # force un noyau impair
    return cv2.GaussianBlur(frame, (k, k), 0)

def f_gris(frame):
    # YOLO attend 3 canaux : gris repassé en BGR, dosé par mélange linéaire
    gray_bgr = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1.0 - gray_strength, gray_bgr, gray_strength, 0)

def f_canny(frame):
    edges_bgr = cv2.cvtColor(cv2.Canny(frame, CANNY_LOW, CANNY_LOW * 2), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1.0 - canny_strength, edges_bgr, canny_strength, 0)

def f_seuil(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1.0 - seuil_strength, binary_bgr, seuil_strength, 0)


def apply_filter(image, max_shift):

    height, width = image.shape[:2]

    # Image noire de sortie
    result = np.zeros_like(image)

    # Parcours de chaque ligne
    for y in range(height):

        # Chaque ligne reçoit un décalage aléatoire
        # entre -max_shift et +max_shift
        shift = np.random.randint(
            -max_shift,
            max_shift + 1
        )

        # Décalage vers la droite
        if shift > 0:

            result[y, shift:] = image[y, :-shift]

        # Décalage vers la gauche
        elif shift < 0:

            result[y, :shift] = image[y, -shift:]

        # Aucun décalage
        else:

            result[y] = image[y]

    return result



def f_decalage(frame):
    if shift_px <= 0:
        return frame
    return apply_filter(frame, shift_px)

FILTERS = [
    ("Flou gaussien", f_flou),
    ("Gris", f_gris),
    ("Canny", f_canny),
    ("Seuil", f_seuil),
    ("Décalage", f_decalage)
]
filter_idx = 0

def intensity_text():
    if filter_idx == 0:
        return f" (k={blur_k})"
    if filter_idx == 1:
        return f" ({int(round(gray_strength * 100))} %)"
    if filter_idx == 2:
        return f" ({int(round(canny_strength * 100))} %)"
    if filter_idx == 3:
        return f" ({int(round(seuil_strength * 100))} %)"
    if filter_idx == 4:
        return f" (±{shift_px} px)"
    return ""

def adjust_intensity(direction):
    # direction : +1 (HAUT, plus d'effet) / -1 (BAS, moins d'effet)
    global blur_k, gray_strength, canny_strength, seuil_strength, shift_px
    if filter_idx == 0:
        blur_k = min(51, max(1, blur_k + 2 * direction))
        print(f"Flou gaussien (k={blur_k})")
    elif filter_idx == 1:
        gray_strength = min(1.0, max(0.0, round(gray_strength + 0.1 * direction, 1)))
        print(f"Gris ({int(round(gray_strength * 100))} %)")
    elif filter_idx == 2:
        canny_strength = min(1.0, max(0.0, round(canny_strength + 0.1 * direction, 1)))
        print(f"Canny ({int(round(canny_strength * 100))} %)")
    elif filter_idx == 3:
        seuil_strength = min(1.0, max(0.0, round(seuil_strength + 0.1 * direction, 1)))
        print(f"Seuil ({int(round(seuil_strength * 100))} %)")
    elif filter_idx == 4:
        shift_px = min(50, max(0, shift_px + 2 * direction))
        print(f"Décalage (±{shift_px} px)")

# waitKeyEx : flèches visibles seulement SANS le masque & 0xFF.
# Codes constatés (Windows/Linux) : gauche=2424832, haut=2490368,
# droite=2555904, bas=2621440. 81/82/83/84 = variante selon builds.
PREV_KEYS = {2424832, 81, ord('a')}
NEXT_KEYS = {2555904, 83, ord('d')}
UP_KEYS = {2490368, 82, ord('w'), ord('z')}  # 'z' = AZERTY (position physique de 'w')
DOWN_KEYS = {2621440, 84, ord('s')}

IMG_SIZE = 320  # inférence réduite comme recepteur.py : ~4x moins de pixels que 640
fps = 0.0
fps_frames = 0
fps_t0 = time.monotonic()
last_mismatch_shape = None  # anti-spam du warning de redimensionnement

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERREUR : Impossible de lire l'image depuis la webcam.")
        break

    filter_name, filter_fn = FILTERS[filter_idx]
    res_left = model(frame, conf=0.7, imgsz=IMG_SIZE, verbose=False)[0].plot()

    filtered_frame = filter_fn(frame)
    res_right = model(filtered_frame, conf=0.7, imgsz=IMG_SIZE, verbose=False)[0].plot()
    cv2.putText(res_right, f"Filtre [{filter_idx + 1}/{len(FILTERS)}] : {filter_name}{intensity_text()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # FPS : fenêtre glissante d'1 s, affiché sur l'image combinée (coût négligeable)
    fps_frames += 1
    now = time.monotonic()
    if now - fps_t0 >= 1.0:
        fps = fps_frames / (now - fps_t0)
        fps_frames = 0
        fps_t0 = now
    if res_left.shape != res_right.shape:
        # Sécurité affichage : la taille de capture n'est pas contractuelle.
        if res_right.shape != last_mismatch_shape:
            print(f"⚠ Tailles incohérentes ({res_left.shape} vs {res_right.shape}) : redimensionnement.")
            last_mismatch_shape = res_right.shape
        res_right = cv2.resize(res_right, (res_left.shape[1], res_left.shape[0]))
    combined = np.hstack((res_left, res_right))
    cv2.putText(combined, f"{fps:.1f} FPS",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("YOLOv11 - Gauche: Brut | Droite: Filtre", combined)

    # Un seul appel par frame : le masque & 0xFF détruit les flèches, donc on
    # teste d'abord le code complet (flèches), puis l'octet bas (lettres).
    key = cv2.waitKeyEx(1)
    if key == -1:
        continue
    if key in PREV_KEYS:
        filter_idx = (filter_idx - 1) % len(FILTERS)
        print(f"Filtre : {FILTERS[filter_idx][0]}{intensity_text()}")
    elif key in NEXT_KEYS:
        filter_idx = (filter_idx + 1) % len(FILTERS)
        print(f"Filtre : {FILTERS[filter_idx][0]}{intensity_text()}")
    elif key in UP_KEYS:
        adjust_intensity(+1)
    elif key in DOWN_KEYS:
        adjust_intensity(-1)
    elif key & 0xFF in (ord('q'), 27):  # 'q' ou Echap
        break

cap.release()
cv2.destroyAllWindows()
print("Programme terminé.")
