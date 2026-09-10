import cv2
import numpy as np
from ultralytics import YOLO

print("1. Chargement du modèle YOLOv11...")
# Au 1er lancement, la ligne suivante télécharge yolo11n.pt (cela peut prendre quelques secondes)
model = YOLO("yolo11n.pt")

print("2. Tentative d'accès à la webcam...")
# Si 0 ne marche pas, essaie avec 1 ou 2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERREUR : Impossible d'ouvrir la webcam.")
    print("- Vérifie qu'aucune autre application (Teams, Zoom, navigateur) n'utilise la caméra.")
    print("- Essaie d'installer le backend dSHOW sur Windows : cv2.VideoCapture(0, cv2.CAP_DSHOW)")
    print("- Modifie l'index de la caméra : cv2.VideoCapture(1)")
    exit()

print("3. Caméra ouverte ! Appuie sur 'q' dans la fenêtre vidéo pour quitter.")

def apply_filter(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERREUR : Impossible de lire l'image depuis la webcam.")
        break

    res_left = model(frame, conf=0.7, verbose=False)[0].plot()

    filtered_frame = apply_filter(frame)
    res_right = model(filtered_frame, conf=0.7, verbose=False)[0].plot()

    cv2.imshow("YOLOv11 - Gauche: Brut | Droite: Filtre", np.hstack((res_left, res_right)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Programme terminé.")