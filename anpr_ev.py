"""
Acuan ANPR + cek EV (lis biru) + voting multi-frame.

Install:
    pip install fast-alpr[onnx] opencv-python      # CPU
    pip install fast-alpr[onnx-gpu] opencv-python  # NVIDIA GPU

Pakai:
    python anpr_ev.py --image foto.jpg
    python anpr_ev.py --folder foto_plat/          # tes banyak foto
    python anpr_ev.py --folder foto_plat/ --csv hasil.csv --fast
    python anpr_ev.py --video 0                    # webcam
    python anpr_ev.py --video rtsp://user:pass@ip/stream

CATATAN: nama field hasil fast-alpr (detection.bounding_box, ocr.text)
dari ingatan; cek README/docs fast-alpr kalau error.
"""
import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fast_alpr import ALPR

# Format plat Indonesia: 1-2 huruf, 1-4 angka, 1-3 huruf (tanpa spasi)
PLATE_RE = re.compile(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$")

# Koreksi karakter ketuker berdasarkan posisi
TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

# HSV biru: KALIBRASI pakai foto plat EV asli
BLUE_LO, BLUE_HI = (95, 80, 90), (130, 255, 255)  # S>=80 & V>=90: abaikan piksel gelap yang cuma kebiruan
BLUE_RATIO_EV = 0.20     # di atas ini = EV
BLUE_RATIO_GRAY = 0.08   # antara GRAY dan EV = ragu
SAVE_DIR = None          # diisi lewat --save-blue buat debug crop + mask biru


def normalize_plate(raw: str):
    """Coba semua cara pecah (awalan, angka, akhiran), pilih yang
    butuh koreksi paling sedikit, lalu validasi regex.

    Contoh: 'B67I6VR2' -> 'B6716VRZ' (I->1, 2->Z menurut posisi).
    BATASAN: kalau OCR sudah membaca huruf yang valid di posisi huruf
    (mis. 'BG543ABC'), fungsi ini menerimanya apa adanya karena 'BG' adalah
    kode wilayah yang sah. Teks saja nggak bisa bedain dengan 'B 6543 ABC'.
    Buat kasus itu perlu confidence per karakter atau fine-tuning OCR.
    """
    t = re.sub(r"[^A-Z0-9]", "", raw.upper())
    best, best_cost = None, 99
    for p in (1, 2):                  # panjang kode wilayah
        for d in (4, 3, 2, 1):        # panjang angka (4 dicoba duluan)
            s = len(t) - p - d        # panjang akhiran
            if not 1 <= s <= 3:
                continue
            cost, out = 0, []
            for seg, want_digit in ((t[:p], False), (t[p:p + d], True), (t[p + d:], False)):
                for ch in seg:
                    if ch.isdigit() == want_digit:
                        out.append(ch)
                    else:
                        m = TO_DIGIT if want_digit else TO_LETTER
                        if ch not in m:
                            cost = 99
                            break
                        out.append(m[ch])
                        cost += 1
                if cost >= 99:
                    break
            if cost < best_cost:
                best, best_cost = "".join(out), cost
    return best if best and PLATE_RE.match(best) else None


def pretty(plate: str) -> str:
    """B6543ABC -> 'B 6543 ABC' (hanya untuk tampilan)."""
    m = PLATE_RE.match(plate)
    return " ".join(m.groups()) if m else plate


def blue_ratio(frame, box, pad_bottom=0.25):
    """Rasio biru TERTINGGI di salah satu pita horizontal di separuh bawah
    plat. Cocok buat lis/corak biru yang nggak memenuhi seluruh strip.
    Box diperbesar ke bawah karena kotak detektor sering memotong lis."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    y2p = min(frame.shape[0], int(y2 + h * pad_bottom))
    crop = frame[max(0, y1):y2p, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, BLUE_LO, BLUE_HI) > 0).astype(np.float32)

    H = mask.shape[0]
    band = max(2, int(H * 0.15))           # tinggi pita ~15% crop
    start = int(H * 0.45)                   # cek dari tengah ke bawah
    step = max(1, band // 2)
    best = 0.0
    for y in range(start, max(start + 1, H - band + 1), step):
        best = max(best, float(mask[y:y + band].mean()))
    if SAVE_DIR:
        d = Path(SAVE_DIR)
        d.mkdir(exist_ok=True)
        n = len(list(d.glob("*_crop.png")))
        cv2.imwrite(str(d / f"{n:03d}_crop.png"), crop)
        cv2.imwrite(str(d / f"{n:03d}_mask.png"), (mask * 255).astype(np.uint8))
    return best


def ev_label(ratio):
    if ratio >= BLUE_RATIO_EV:
        return "EV"
    if ratio >= BLUE_RATIO_GRAY:
        return "RAGU"
    return "NON_EV"


class Voter:
    """Gabung hasil beberapa frame: plat dengan total confidence tertinggi."""

    def __init__(self):
        self.score = defaultdict(float)
        self.ratios = defaultdict(list)

    def add(self, plate, conf, ratio):
        self.score[plate] += conf
        self.ratios[plate].append(ratio)

    def best(self):
        if not self.score:
            return None
        plate = max(self.score, key=self.score.get)
        return plate, float(np.median(self.ratios[plate]))


ANGLES = (25, -25, 15, -15, 35, -35)   # sudut coba-coba buat plat miring


def rotate(img, angle):
    """Putar gambar (derajat, + = berlawanan arah jarum jam) tanpa memotong sisi."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(127, 127, 127))


def _mean_conf(conf):
    if isinstance(conf, (list, tuple, np.ndarray)):
        return float(np.mean(conf))
    return float(conf)


def _predict(alpr, frame, debug=False):
    """Satu kali deteksi + OCR. Return list (plate, conf, blue_ratio)."""
    out = []
    for r in alpr.predict(frame):
        if r.ocr is None:
            continue
        plate = normalize_plate(r.ocr.text)
        if debug:
            print(f"  [DEBUG] mentah={r.ocr.text!r} -> {plate}")
        if plate is None:
            continue
        bb = r.detection.bounding_box
        box = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2))
        out.append((plate, _mean_conf(r.ocr.confidence), blue_ratio(frame, box)))
    return out


def _ocr_only(alpr, frame, debug=False):
    """Cadangan terakhir: lewati detektor, anggap seluruh gambar = plat."""
    try:
        o = alpr.ocr.predict(frame)
    except Exception as e:  # nama API bisa beda antar versi fast-alpr
        if debug:
            print(f"  [DEBUG] OCR-only gagal: {e}")
        return []
    if o is None:
        return []
    plate = normalize_plate(o.text)
    if debug:
        print(f"  [DEBUG] OCR-only mentah={o.text!r} -> {plate}")
    if plate is None:
        return []
    h, w = frame.shape[:2]
    return [(plate, _mean_conf(o.confidence), blue_ratio(frame, (0, 0, w, h)))]


def _score(c):
    """Skor kandidat = confidence x panjang plat (~ total confidence tiap karakter).
    Bacaan sebagian ('B6595J') kalah dari bacaan lengkap ('B6595JFZ')."""
    return c[1] * len(c[0])


def read_frame(alpr, frame, debug=False, deep=False):
    """Return list of (plate, conf, blue_ratio) untuk satu frame.

    deep=True (default buat foto): SELALU coba beberapa varian gambar
    (asli, +bingkai kosong, +bingkai & diputar), lalu pilih kandidat
    dengan skor tertinggi. Ini nangkep plat miring / kepotong di kotak
    detektor. Kalau semua gagal, OCR langsung tanpa detektor.
    Hasil deep = 1 plat terbaik. Video tetap deep=False biar cepat.
    """
    res = _predict(alpr, frame, debug)
    if not deep:
        return res

    cands = [("asli", c) for c in res]
    h, w = frame.shape[:2]
    pad = int(0.4 * max(h, w))
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(127, 127, 127))
    tries = [("bingkai", padded)]
    tries += [(f"bingkai+putar {a}", rotate(padded, a)) for a in ANGLES]
    for name, img in tries:
        cands += [(name, c) for c in _predict(alpr, img, debug)]

    if not cands:
        for a in (0,) + ANGLES:
            img = rotate(frame, a) if a else frame
            cands += [(f"OCR-only putar {a}", c) for c in _ocr_only(alpr, img, debug)]
            if cands:
                break
    if not cands:
        return []

    name, best = max(cands, key=lambda x: _score(x[1]))
    if debug:
        for n, c in cands:
            print(f"  [DEBUG] kandidat {n}: {c[0]} conf={c[1]:.2f} skor={_score(c):.2f}")
        print(f"  [DEBUG] dipilih dari: {name}")
    return [best]


def run_folder(alpr, folder, debug=False, deep=True, csv_path=None):
    rows = []
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p.name}: GAMBAR RUSAK / TIDAK BISA DIBUKA")
            rows.append([p.name, "", "", "", "RUSAK", "", ""])
            continue
        res = read_frame(alpr, img, debug, deep=deep)
        if not res:
            print(f"{p.name}: TIDAK TERBACA")
            rows.append([p.name, "", "", "", "TIDAK_TERBACA", "", ""])
            continue
        plate, conf, ratio = max(res, key=lambda x: x[1])
        label = ev_label(ratio)
        print(f"{p.name}: {pretty(plate)} conf={conf:.2f} biru={ratio:.2f} -> {label}")
        rows.append([p.name, plate, f"{conf:.3f}", f"{ratio:.3f}", label, "", ""])

    if csv_path:
        # utf-8-sig biar kebuka rapi di Excel. Dua kolom terakhir diisi manual
        # (plat_benar, ev_benar = EV / NON_EV) lalu dipakai evaluate.py
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "plat", "conf", "biru", "label", "plat_benar", "ev_benar"])
            w.writerows(rows)
        print(f"\nCSV tersimpan: {csv_path} ({len(rows)} baris)")


def run_video(alpr, src, window_s=2.0, debug=False):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    voter, t0 = Voter(), time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        for plate, conf, ratio in read_frame(alpr, frame, debug):
            voter.add(plate, conf, ratio)
        if time.time() - t0 >= window_s:
            best = voter.best()
            if best:
                plate, ratio = best
                print(f"HASIL: {pretty(plate)} | biru={ratio:.2f} -> {ev_label(ratio)}")
            voter, t0 = Voter(), time.time()
    cap.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--folder")
    ap.add_argument("--video")
    ap.add_argument("--csv", metavar="FILE",
                    help="(dengan --folder) simpan hasil ke CSV, mis. hasil.csv")
    ap.add_argument("--fast", action="store_true",
                    help="foto: lewati pencarian bertahap (lebih cepat, kurang teliti)")
    ap.add_argument("--debug", action="store_true",
                    help="tampilkan teks OCR mentah vs hasil koreksi")
    ap.add_argument("--save-blue", metavar="DIR",
                    help="simpan crop plat + mask biru ke folder DIR (buat kalibrasi)")
    a = ap.parse_args()
    SAVE_DIR = a.save_blue

    alpr = ALPR()  # model default; lihat README buat ganti model
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            print("Gambar tidak bisa dibuka:", a.image)
        else:
            for plate, conf, ratio in read_frame(alpr, img, a.debug, deep=not a.fast):
                print(f"{pretty(plate)} conf={conf:.2f} biru={ratio:.2f} -> {ev_label(ratio)}")
    elif a.folder:
        run_folder(alpr, a.folder, a.debug, deep=not a.fast, csv_path=a.csv)
    elif a.video:
        run_video(alpr, a.video, debug=a.debug)
    else:
        ap.print_help()