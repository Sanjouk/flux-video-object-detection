# emeteur.py — capture la webcam locale et envoie chaque frame en UDP vers le recepteur.
# Protocole volontairement simple : 1 frame JPEG = 1 datagramme UDP (pas de re-assemblage).
# A lancer sur la machine avec la camera ; le recepteur ecoute sur PORT.
import cv2
import socket
import errno

# 1. Utilise l'IP directe pour tester d'abord si le flux passe
TARGET_IP = "10.77.180.161"
PORT = 5000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 40
MAX_FRAME_BYTES = 60000  # Marge sous la limite d'un datagramme UDP IPv4.

# Socket UDP non connecte : pas de handshake, latence minimale, perte tolerable en video.
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# macOS utilise par defaut un buffer UDP de seulement 9216 octets.
# Un JPEG peut respecter la limite UDP et depasser quand meme ce buffer.
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
max_frame_bytes = min(MAX_FRAME_BYTES, sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
# Autorise le broadcast au cas ou TARGET_IP serait une adresse de diffusion (ex. 192.168.1.255).
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Ouverture de la webcam par defaut (index 0).
cap = cv2.VideoCapture(0)

# Resolution demandee ; certaines cameras peuvent choisir un autre mode.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
print(
    f"Résolution caméra : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
)

print(f"Début de l'envoi vers {TARGET_IP}:{PORT}...")

# Boucle principale : lit, compresse, envoie — tant que la camera fournit des frames.
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # Echec de lecture (camera debranchee/fin de flux) : on sort pour liberer plus bas.
        break

    # Si le JPEG est trop gros, reduit la qualite pour garder 1 frame = 1 paquet.
    data = None
    for quality in range(JPEG_QUALITY, 19, -10):
        encoded, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not encoded:
            break
        if buffer.nbytes <= max_frame_bytes:
            data = buffer.tobytes()
            break
    if data is None:
        # Frame invalide ou encore trop grosse : ne pas envoyer un paquet impossible.
        continue

    # Envoi best-effort : en UDP un paquet peut se perdre, on log et on passe a la suivante.
    try:
        sock.sendto(data, (TARGET_IP, PORT))
    except OSError as e:
        if e.errno == errno.EMSGSIZE:
            # Adapte les prochains JPEG si le chemin reseau impose une limite inferieure.
            max_frame_bytes = max(1, len(data) * 3 // 4)
            print(f"Paquet trop grand ({len(data)} octets), limite JPEG réduite à {max_frame_bytes} octets.")
        else:
            print(f"Erreur d'envoi : {e}")

# Liberation camera + socket, meme apres une sortie prematuree de la boucle.
cap.release()
sock.close()
