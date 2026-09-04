# emeteur.py — capture la webcam locale et envoie chaque frame en UDP vers le recepteur.
# Protocole volontairement simple : 1 frame JPEG = 1 datagramme UDP (pas de re-assemblage).
# A lancer sur la machine avec la camera ; le recepteur ecoute sur PORT.
import cv2
import socket

# 1. Utilise l'IP directe pour tester d'abord si le flux passe
TARGET_IP = '10.18.5.147'  
PORT = 5000

# Socket UDP non connecte : pas de handshake, latence minimale, perte tolerable en video.
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Autorise le broadcast au cas ou TARGET_IP serait une adresse de diffusion (ex. 192.168.1.255).
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Ouverture de la webcam par defaut (index 0).
cap = cv2.VideoCapture(0)

# Resolution volontairement basse : une frame doit tenir dans un seul datagramme UDP (< 64 Ko).
# Réduction de la résolution à 480x360 pour alléger drastiquement les paquets UDP
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print(f"Début de l'envoi vers {TARGET_IP}:{PORT}...")

# Boucle principale : lit, compresse, envoie — tant que la camera fournit des frames.
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # Echec de lecture (camera debranchee/fin de flux) : on sort pour liberer plus bas.
        break

    # Compression JPEG basse qualite : compromis poids reseau (~15-20 Ko) / lisibilite pour YOLO.
    # Qualité 40% pour garantir un paquet sous les 15-20 Ko
    encoded, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    if not encoded:
        # Encodage echoue : on saute la frame plutot que d'envoyer des donnees invalides.
        continue

    # Conversion en bytes : c'est ce payload qui voyage dans le datagramme UDP.
    data = buffer.tobytes()

    # Envoi best-effort : en UDP un paquet peut se perdre, on log et on passe a la suivante.
    try:
        sock.sendto(data, (TARGET_IP, PORT))
    except Exception as e:
        print(f"Erreur d'envoi : {e}")

# Liberation camera + socket, meme apres une sortie prematuree de la boucle.
cap.release()
sock.close()
