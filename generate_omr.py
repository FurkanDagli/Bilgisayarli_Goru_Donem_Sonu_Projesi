"""
generate_omr.py - Optik Cevap Formu Şablon Üretici
Programatik olarak OMR (Optical Mark Recognition) formları üretir.
Cevap anahtarı ve rastgele doldurulmuş öğrenci formları oluşturur.
"""

import cv2
import numpy as np
import os
import random

# ============================================================
# OMR Form Düzeni Sabitleri
# Bu sabitler bolum2_optik.py tarafından da kullanılır.
# ============================================================

PAGE_W = 800
PAGE_H = 1100

# Köşe işaretleri (hizalama için)
MARKER_SIZE = 25
MARKERS = {
    "tl": (30, 30),                          # sol-üst (x, y)
    "tr": (PAGE_W - 30 - MARKER_SIZE, 30),   # sağ-üst
    "bl": (30, PAGE_H - 30 - MARKER_SIZE),   # sol-alt
    "br": (PAGE_W - 30 - MARKER_SIZE, PAGE_H - 30 - MARKER_SIZE),  # sağ-alt
}

# Köşe merkezleri (perspektif düzeltme referansları)
MARKER_CENTERS = {
    k: (v[0] + MARKER_SIZE // 2, v[1] + MARKER_SIZE // 2)
    for k, v in MARKERS.items()
}

# Öğrenci numarası bölgesi
STUDENT_ID = {
    "origin": (180, 175),    # İlk baloncuk merkezi (hane 1, rakam 0)
    "cols": 9,               # 9 haneli numara
    "rows": 10,              # 0-9 rakamları
    "h_spacing": 48,         # Sütunlar arası mesafe
    "v_spacing": 28,         # Satırlar arası mesafe
    "bubble_radius": 10,
}

# Cevap bölgesi
ANSWERS = {
    "col1_origin": (150, 555),   # 1. sütun ilk baloncuk (Soru 1, A)
    "col2_origin": (500, 555),   # 2. sütun ilk baloncuk (Soru 11, A)
    "questions_per_col": 10,
    "options": 5,                # A, B, C, D, E
    "h_spacing": 45,             # Seçenekler arası mesafe
    "v_spacing": 36,             # Sorular arası mesafe
    "bubble_radius": 10,
}

NUM_QUESTIONS = 20
OPTION_LABELS = ["A", "B", "C", "D", "E"]


def _draw_markers(img):
    """4 köşeye hizalama işaretlerini çizer."""
    for key, (x, y) in MARKERS.items():
        cv2.rectangle(img, (x, y), (x + MARKER_SIZE, y + MARKER_SIZE), 0, -1)


def _draw_header(img):
    """Form başlığını çizer."""
    cv2.putText(img, "OPTIK CEVAP FORMU", (220, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    # Alt çizgi
    cv2.line(img, (220, 85), (600, 85), 0, 1)


def _draw_student_id_section(img):
    """Öğrenci numarası bölgesini çizer (boş baloncuklar)."""
    cfg = STUDENT_ID
    ox, oy = cfg["origin"]

    # Bölge başlığı
    cv2.putText(img, "Ogrenci No:", (60, oy - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)

    # Hane başlıkları (H1, H2, ... H9)
    for col in range(cfg["cols"]):
        cx = ox + col * cfg["h_spacing"]
        cv2.putText(img, str(col + 1), (cx - 5, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)

    # Satır etiketleri (0-9) ve baloncuklar
    for row in range(cfg["rows"]):
        cy = oy + row * cfg["v_spacing"]
        # Rakam etiketi
        cv2.putText(img, str(row), (ox - 30, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)
        for col in range(cfg["cols"]):
            cx = ox + col * cfg["h_spacing"]
            cv2.circle(img, (cx, cy), cfg["bubble_radius"], 0, 1)


def _draw_answer_section(img):
    """Cevap bölgesini çizer (boş baloncuklar)."""
    cfg = ANSWERS

    # Ayırıcı çizgi
    cv2.line(img, (60, 520), (740, 520), 0, 1)

    # Bölge başlığı
    cv2.putText(img, "Cevaplar:", (60, 545),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)

    for col_idx, origin in enumerate([cfg["col1_origin"], cfg["col2_origin"]]):
        ox, oy = origin

        # Seçenek başlıkları (A, B, C, D, E)
        for opt in range(cfg["options"]):
            lx = ox + opt * cfg["h_spacing"]
            cv2.putText(img, OPTION_LABELS[opt], (lx - 5, oy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)

        for row in range(cfg["questions_per_col"]):
            q_num = col_idx * cfg["questions_per_col"] + row + 1
            cy = oy + row * cfg["v_spacing"]

            # Soru numarası
            label = f"{q_num}."
            label_x = ox - 40 if q_num < 10 else ox - 48
            cv2.putText(img, label, (label_x, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)

            for opt in range(cfg["options"]):
                cx = ox + opt * cfg["h_spacing"]
                cv2.circle(img, (cx, cy), cfg["bubble_radius"], 0, 1)


def generate_blank_form():
    """Boş bir OMR formu oluşturur.

    Returns:
        Beyaz arka planlı, boş baloncuklu OMR formu (gri tonlama).
    """
    img = np.ones((PAGE_H, PAGE_W), dtype=np.uint8) * 255

    _draw_markers(img)
    _draw_header(img)
    _draw_student_id_section(img)
    _draw_answer_section(img)

    return img


def fill_bubble(img, center, radius, fill_level=0.85):
    """Bir baloncuğu doldurur.

    Args:
        img: Form görüntüsü.
        center: Baloncuk merkezi (x, y).
        radius: Baloncuk yarıçapı.
        fill_level: Doluluuk oranı (0-1). 1.0 = tam siyah.
    """
    intensity = int(255 * (1 - fill_level))
    cv2.circle(img, center, radius - 1, intensity, -1)


def fill_student_id(img, student_id_str):
    """Öğrenci numarasını forma işler.

    Args:
        img: Form görüntüsü.
        student_id_str: 9 haneli öğrenci numarası (string).
    """
    cfg = STUDENT_ID
    ox, oy = cfg["origin"]

    student_id_str = student_id_str.zfill(9)[:9]

    for col, digit_char in enumerate(student_id_str):
        digit = int(digit_char)
        cx = ox + col * cfg["h_spacing"]
        cy = oy + digit * cfg["v_spacing"]
        fill_level = random.uniform(0.75, 0.95)
        fill_bubble(img, (cx, cy), cfg["bubble_radius"], fill_level)


def fill_answers(img, answers):
    """Cevapları forma işler.

    Args:
        img: Form görüntüsü.
        answers: 20 elemanlı liste. Her eleman 0-4 arası int (A=0, B=1, ...)
                 veya None (boş bırakılmış soru).
    """
    cfg = ANSWERS

    for q_idx, ans in enumerate(answers):
        if ans is None:
            continue

        if q_idx < cfg["questions_per_col"]:
            ox, oy = cfg["col1_origin"]
            row = q_idx
        else:
            ox, oy = cfg["col2_origin"]
            row = q_idx - cfg["questions_per_col"]

        cx = ox + ans * cfg["h_spacing"]
        cy = oy + row * cfg["v_spacing"]
        fill_level = random.uniform(0.75, 0.95)
        fill_bubble(img, (cx, cy), cfg["bubble_radius"], fill_level)


def add_noise(img, rotation_range=(-2, 2), noise_std=5):
    """Gerçekçi gürültü ve hafif dönüklük ekler.

    Args:
        img: Form görüntüsü.
        rotation_range: Rastgele dönüş açısı aralığı (derece).
        noise_std: Gaussian gürültü standart sapması.

    Returns:
        Gürültülü form görüntüsü.
    """
    result = img.copy()

    # Gaussian gürültü
    noise = np.random.normal(0, noise_std, result.shape).astype(np.float32)
    result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Hafif dönüklük
    angle = random.uniform(*rotation_range)
    h, w = result.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)

    # Hafif bulanıklaştırma (tarama efekti)
    if random.random() > 0.5:
        result = cv2.GaussianBlur(result, (3, 3), 0)

    return result


def generate_test_set(output_dir="test_images/bolum2", n_students=10):
    """Tam bir test seti üretir: cevap anahtarı + n adet öğrenci formu.

    Args:
        output_dir: Çıktı klasörü.
        n_students: Üretilecek öğrenci formu sayısı.

    Returns:
        (answer_key, student_data) tuple'ı.
        answer_key: 20 elemanlı int listesi (doğru cevaplar, 0-4).
        student_data: [{id, answers, filled_answers}] listesi.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Cevap Anahtarı ---
    answer_key = [random.randint(0, 4) for _ in range(NUM_QUESTIONS)]

    # Cevap anahtarı formunu oluştur
    key_form = generate_blank_form()
    fill_student_id(key_form, "000000000")
    fill_answers(key_form, answer_key)
    key_path = os.path.join(output_dir, "cevap_anahtari.png")
    cv2.imwrite(key_path, key_form)
    print(f"[BILGI] Cevap anahtari kaydedildi: {key_path}")

    # Cevap anahtarını metin dosyasına da yaz
    key_txt_path = os.path.join(output_dir, "cevap_anahtari.txt")
    with open(key_txt_path, "w", encoding="utf-8") as f:
        f.write("# Cevap Anahtari\n")
        f.write("# Soru: Cevap\n")
        for i, ans in enumerate(answer_key):
            f.write(f"{i + 1}: {OPTION_LABELS[ans]}\n")
    print(f"[BILGI] Cevap anahtari metni: {key_txt_path}")

    # --- Öğrenci Formları ---
    student_data = []
    for s in range(n_students):
        student_id = f"{random.randint(100000000, 999999999)}"

        # Rastgele cevaplar üret (bazılarını boş bırak)
        answers = []
        for q in range(NUM_QUESTIONS):
            r = random.random()
            if r < 0.1:  # %10 boş
                answers.append(None)
            else:
                answers.append(random.randint(0, 4))

        # Formu oluştur ve doldur
        form = generate_blank_form()
        fill_student_id(form, student_id)
        fill_answers(form, answers)

        # Gürültü ekle
        form_noisy = add_noise(form)

        # Kaydet
        filename = f"ogrenci_{s + 1:02d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, form_noisy)

        student_data.append({
            "id": student_id,
            "answers": answers,
            "filename": filename,
        })

        print(f"  Ogrenci {s + 1:2d}: No={student_id}, "
              f"Dosya={filename}")

    # Beklenen sonuçları kaydet (doğrulama için)
    expected_path = os.path.join(output_dir, "beklenen_sonuclar.txt")
    with open(expected_path, "w", encoding="utf-8") as f:
        f.write("# Beklenen Sonuclar (Dogrulama icin)\n")
        f.write(f"# Cevap Anahtari: {[OPTION_LABELS[a] for a in answer_key]}\n\n")
        for sd in student_data:
            dogru = sum(1 for q, a in enumerate(sd["answers"])
                        if a is not None and a == answer_key[q])
            yanlis = sum(1 for q, a in enumerate(sd["answers"])
                         if a is not None and a != answer_key[q])
            bos = sum(1 for a in sd["answers"] if a is None)
            f.write(f"Ogrenci: {sd['id']} | "
                    f"Dogru: {dogru:2d} | Yanlis: {yanlis:2d} | Bos: {bos:2d}\n")

    print(f"\n[BILGI] Beklenen sonuclar: {expected_path}")
    print(f"[TAMAM] {n_students} ogrenci formu uretildi -> {output_dir}")

    return answer_key, student_data


if __name__ == "__main__":
    print("=" * 60)
    print("  OPTIK FORM URETICI")
    print("=" * 60)
    generate_test_set()
