"""
bolum2_optik.py - Optik Form Okuyucu
Bölüm 2: Cevap anahtarı ile öğrenci formlarını okuyup değerlendirir.
"""

import cv2
import numpy as np
import os

from generate_omr import (
    PAGE_W, PAGE_H, MARKER_SIZE, MARKER_CENTERS,
    STUDENT_ID, ANSWERS, NUM_QUESTIONS, OPTION_LABELS
)
from utils import four_point_transform


def find_corner_markers(image):
    """Form üzerindeki 4 köşe işaretini bulur.

    Siyah kare şeklindeki hizalama işaretlerini tespit eder.

    Args:
        image: Giriş görüntüsü (gri tonlama veya BGR).

    Returns:
        4 köşe merkez noktası (4, 2) float32 dizisi
        veya bulunamazsa None.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binary dönüşüm
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Konturları bul
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        # Marker boyutuna yakın alanlar (toleranslı)
        expected_area = MARKER_SIZE * MARKER_SIZE
        if not (expected_area * 0.3 < area < expected_area * 3.0):
            continue

        # Kareye yakın olmalı
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        if not (0.6 < aspect < 1.4):
            continue

        # Merkez noktası
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        candidates.append((cx, cy))

    if len(candidates) < 4:
        return None

    # En yakın 4 köşeyi bul
    corners = {}
    for name, target in [("tl", (0, 0)), ("tr", (w, 0)),
                          ("bl", (0, h)), ("br", (w, h))]:
        best = None
        best_dist = float("inf")
        for pt in candidates:
            dist = np.sqrt((pt[0] - target[0]) ** 2 + (pt[1] - target[1]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = pt
        corners[name] = best

    pts = np.array([corners["tl"], corners["tr"],
                    corners["br"], corners["bl"]], dtype=np.float32)
    return pts


def align_form(image):
    """Formu hizalar (perspektif düzeltme).

    Köşe işaretlerini bulup, formu kanonik boyutlara (PAGE_W x PAGE_H)
    dönüştürür.

    Args:
        image: Giriş görüntüsü.

    Returns:
        Hizalanmış form görüntüsü (PAGE_W x PAGE_H) veya None.
    """
    corners = find_corner_markers(image)

    if corners is None:
        print("[UYARI] Kose isaretleri bulunamadi, dogrudan boyutlandirma yapiliyor.")
        # Fallback: doğrudan boyutlandır
        resized = cv2.resize(image, (PAGE_W, PAGE_H))
        if len(resized.shape) == 3:
            return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return resized

    # Hedef noktalar (marker merkezleri)
    dst = np.array([
        list(MARKER_CENTERS["tl"]),
        list(MARKER_CENTERS["tr"]),
        list(MARKER_CENTERS["br"]),
        list(MARKER_CENTERS["bl"]),
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    aligned = cv2.warpPerspective(image, M, (PAGE_W, PAGE_H))

    if len(aligned.shape) == 3:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    return aligned


def read_bubble_grid(image, origin, rows, cols, h_spacing, v_spacing, radius):
   
    ox, oy = origin
    grid = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            cx = ox + c * h_spacing
            cy = oy + r * v_spacing

            # Baloncuk bölgesini kırp
            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(image.shape[1], cx + radius)
            y2 = min(image.shape[0], cy + radius)

            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                grid[r, c] = 255
            else:
                grid[r, c] = np.mean(roi)

    return grid


def read_student_id(image, debug_image=None):
    """Öğrenci numarasını okur.

    Args:
        image: Hizalanmış gri tonlamalı form görüntüsü.

    Returns:
        9 haneli öğrenci numarası (string).
    """
    cfg = STUDENT_ID
    grid = read_bubble_grid(
        image,
        origin=cfg["origin"],
        rows=cfg["rows"],
        cols=cfg["cols"],
        h_spacing=cfg["h_spacing"],
        v_spacing=cfg["v_spacing"],
        radius=cfg["bubble_radius"]
    )

    # Her sütunda en koyu (en düşük yoğunluk) baloncuğu seç
    student_id = ""
    for col in range(cfg["cols"]):
        col_values = grid[:, col]
        digit = np.argmin(col_values)
        student_id += str(digit)
        
        # Debug görüntüsüne çiz
        if debug_image is not None:
            cx = cfg["origin"][0] + col * cfg["h_spacing"]
            cy = cfg["origin"][1] + digit * cfg["v_spacing"]
            cv2.circle(debug_image, (cx, cy), cfg["bubble_radius"] + 2, (0, 255, 0), 2)

    return student_id


def read_answers(image, debug_image=None):
    """Cevapları okur.

    Args:
        image: Hizalanmış gri tonlamalı form görüntüsü.

    Returns:
        20 elemanlı liste. Her eleman:
        - 0-4 (A-E) → işaretlenmiş cevap
        - None → boş bırakılmış
    """
    cfg = ANSWERS
    answers = []

    # Doluluk eşiği - bu değerin altındaki yoğunluk "dolu" sayılır
    FILL_THRESHOLD = 180

    for col_idx, origin in enumerate([cfg["col1_origin"], cfg["col2_origin"]]):
        grid = read_bubble_grid(
            image,
            origin=origin,
            rows=cfg["questions_per_col"],
            cols=cfg["options"],
            h_spacing=cfg["h_spacing"],
            v_spacing=cfg["v_spacing"],
            radius=cfg["bubble_radius"]
        )

        for row in range(cfg["questions_per_col"]):
            row_values = grid[row, :]
            min_idx = np.argmin(row_values)
            min_val = row_values[min_idx]

            if min_val < FILL_THRESHOLD:
                answers.append(min_idx)
                
                # Debug görüntüsüne çiz
                if debug_image is not None:
                    cx = origin[0] + min_idx * cfg["h_spacing"]
                    cy = origin[1] + row * cfg["v_spacing"]
                    cv2.circle(debug_image, (cx, cy), cfg["bubble_radius"] + 2, (0, 255, 0), 2)
            else:
                answers.append(None)

    return answers


def grade_paper(image_path, answer_key):
    """Tek bir öğrenci formunu değerlendirir.

    Args:
        image_path: Form görüntüsü dosya yolu.
        answer_key: Doğru cevaplar listesi (20 eleman, her biri 0-4).

    Returns:
        Sonuç sözlüğü: {student_id, answers, dogru, yanlis, bos, puan}
        veya hata durumunda None.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[HATA] Goruntu okunamadi: {image_path}")
        return None

    # Formu hizala
    aligned = align_form(image)
    if aligned is None:
        return None
        
    debug_image = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)

    # Öğrenci numarasını oku
    student_id = read_student_id(aligned, debug_image)

    # Cevapları oku
    answers = read_answers(aligned, debug_image)

    # Değerlendir
    dogru = 0
    yanlis = 0
    bos = 0
    detay = []

    for q in range(NUM_QUESTIONS):
        if answers[q] is None:
            bos += 1
            detay.append(f"  Soru {q + 1:2d}: BOS          "
                         f"(Dogru: {OPTION_LABELS[answer_key[q]]})")
        elif answers[q] == answer_key[q]:
            dogru += 1
            detay.append(f"  Soru {q + 1:2d}: {OPTION_LABELS[answers[q]]}  DOGRU")
        else:
            yanlis += 1
            detay.append(f"  Soru {q + 1:2d}: {OPTION_LABELS[answers[q]]}  YANLIS    "
                         f"(Dogru: {OPTION_LABELS[answer_key[q]]})")

    puan = round(dogru / NUM_QUESTIONS * 100, 1)

    # Debug görüntüsünü kaydet
    base_name = os.path.basename(image_path)
    debug_dir = os.path.join("output", "bolum2")
    os.makedirs(debug_dir, exist_ok=True)
    out_name = base_name.replace(".png", "_isaretli.png").replace(".jpg", "_isaretli.jpg")
    cv2.imwrite(os.path.join(debug_dir, out_name), debug_image)

    return {
        "student_id": student_id,
        "answers": answers,
        "dogru": dogru,
        "yanlis": yanlis,
        "bos": bos,
        "puan": puan,
        "detay": detay,
    }


def load_answer_key_from_file(filepath):
    """Cevap anahtarını metin dosyasından yükler.

    Args:
        filepath: cevap_anahtari.txt dosya yolu.

    Returns:
        20 elemanlı int listesi (0-4).
    """
    answer_key = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(":")
            if len(parts) == 2:
                ans_letter = parts[1].strip().upper()
                if ans_letter in OPTION_LABELS:
                    answer_key.append(OPTION_LABELS.index(ans_letter))
    return answer_key


def batch_grade(folder, answer_key_path=None, answer_key=None):
    """Bir klasördeki tüm öğrenci formlarını toplu değerlendirir.

    Args:
        folder: Öğrenci formlarının bulunduğu klasör.
        answer_key_path: Cevap anahtarı metin dosyası yolu (opsiyonel).
        answer_key: Doğrudan cevap anahtarı listesi (opsiyonel).

    Returns:
        Sonuçlar listesi.
    """
    # Cevap anahtarını yükle
    if answer_key is None and answer_key_path:
        answer_key = load_answer_key_from_file(answer_key_path)
    elif answer_key is None:
        # Klasörde cevap_anahtari.txt ara
        default_path = os.path.join(folder, "cevap_anahtari.txt")
        if os.path.exists(default_path):
            answer_key = load_answer_key_from_file(default_path)
        else:
            print("[HATA] Cevap anahtari bulunamadi!")
            return []

    if len(answer_key) != NUM_QUESTIONS:
        print(f"[HATA] Cevap anahtarinda {len(answer_key)} soru var, "
              f"{NUM_QUESTIONS} bekleniyor!")
        return []

    # Öğrenci formlarını bul
    files = sorted([
        f for f in os.listdir(folder)
        if f.startswith("ogrenci_") and f.endswith(".png")
    ])

    if not files:
        print(f"[UYARI] '{folder}' klasorunde ogrenci formu bulunamadi.")
        return []

    print(f"[BILGI] {len(files)} ogrenci formu bulundu.\n")

    results = []
    for filename in files:
        filepath = os.path.join(folder, filename)
        print(f"--- {filename} ---")
        result = grade_paper(filepath, answer_key)

        if result:
            results.append(result)
            print(f"  Ogrenci No : {result['student_id']}")
            print(f"  Dogru      : {result['dogru']}")
            print(f"  Yanlis     : {result['yanlis']}")
            print(f"  Bos        : {result['bos']}")
            print(f"  Puan       : %{result['puan']}")
            print()

    # Özet tablo
    if results:
        print("\n" + "=" * 70)
        print(f"{'OGRENCI NO':<15} {'DOGRU':>6} {'YANLIS':>7} {'BOS':>5} {'PUAN':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r['student_id']:<15} {r['dogru']:>6} {r['yanlis']:>7} "
                  f"{r['bos']:>5} {r['puan']:>7}%")
        print("-" * 70)

        avg_puan = sum(r["puan"] for r in results) / len(results)
        print(f"{'ORTALAMA':<15} {'':>6} {'':>7} {'':>5} {avg_puan:>7.1f}%")
        print("=" * 70)

    # Sonuçları dosyaya kaydet
    output_path = os.path.join("output", "bolum2", "sonuclar.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'OGRENCI NO':<15} {'DOGRU':>6} {'YANLIS':>7} {'BOS':>5} {'PUAN':>8}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['student_id']:<15} {r['dogru']:>6} {r['yanlis']:>7} "
                    f"{r['bos']:>5} {r['puan']:>7}%\n")
        if results:
            avg_puan = sum(r["puan"] for r in results) / len(results)
            f.write("-" * 70 + "\n")
            f.write(f"{'ORTALAMA':<15} {'':>6} {'':>7} {'':>5} {avg_puan:>7.1f}%\n")
    print(f"\n[BILGI] Sonuclar kaydedildi: {output_path}")

    return results
