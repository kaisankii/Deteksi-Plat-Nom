
import cv2
import time
from fast_alpr import ALPR

# Inisialisasi model satu kali saja
alpr = ALPR()

# Buka webcam Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera gagal dibuka. Coba ganti index 0 menjadi 1.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_inference = 0
inference_interval = 0.7  # detik
detections = []

print("Kamera aktif. Tekan Q pada jendela kamera untuk keluar.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame kamera gagal dibaca.")
        break

    now = time.time()

    # Jalankan ANPR berkala agar preview tetap responsif
    if now - last_inference >= inference_interval:
        try:
            detections = []
            for result in alpr.predict(frame):
                if result.ocr is None:
                    continue

                plate = result.ocr.text
                bb = result.detection.bounding_box
                box = (int(bb.x1), int(bb.y1),
                       int(bb.x2), int(bb.y2))

                detections.append((plate, box))

                print("Plat terbaca:", plate)

        except Exception as e:
            print("ANPR error:", e)

        last_inference = now

    # Gambar bounding box pada preview
    for plate, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, plate, (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0), 2
        )

    cv2.imshow("Live ANPR Camera - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
