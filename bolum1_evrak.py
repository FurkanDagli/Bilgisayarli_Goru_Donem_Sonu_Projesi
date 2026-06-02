"""
bolum1_evrak.py - Evrak/Sayfa Tarama İşlemi
Bölüm 1: Maskeleme, kontrast iyileştirme ve perspektif düzeltme.
"""

import cv2
import numpy as np
import os
from utils import resize_image, order_points, four_point_transform


def detect_document(image):
    """Görüntüdeki evrak/sayfa bölgesini tespit eder.

    Gri tonlama → Bulanıklaştırma → Canny kenar algılama → Kontur tespiti
    ile en büyük 4-köşeli konturu bulur.

    Args:
        image: Giriş görüntüsü (BGR).

    Returns:
        4 köşe noktası (4, 2) float32 dizisi veya bulunamazsa None.
    """
    # İşlem için küçült
    ratio = image.shape[0] / 500.0
    resized = resize_image(image, width=int(image.shape[1] / ratio))

    # Gri tonlama ve bulanıklaştırma
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kenar algılama
    edged = cv2.Canny(blurred, 50, 200)

    # Kenarları güçlendir
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    # Konturları bul
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is not None:
        # Orijinal boyuta ölçekle
        doc_contour = doc_contour.reshape(4, 2).astype(np.float32) * ratio

    return doc_contour


def apply_mask(image, contour):
    """Evrak dışındaki bölgeyi maskeler, arka planı beyaz yapar.

    Args:
        image: Giriş görüntüsü (BGR).
        contour: Evrak köşe noktaları (4, 2).

    Returns:
        Maskelenmiş görüntü (evrak dışı beyaz).
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    contour_int = contour.reshape(4, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, [contour_int], 255)

    # Evrak bölgesini al
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Arka planı beyaz yap
    bg = np.ones_like(image, dtype=np.uint8) * 255
    bg_mask = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(bg, bg, mask=bg_mask)
    result = cv2.add(masked, bg)

    return result


def enhance_contrast(image):
    """Kontrast iyileştirme ve zemin beyazlatma.

    CLAHE ile kontrast artırma, adaptive thresholding ile
    metin-zemin ayrımı ve morfolojik temizlik uygular.

    Args:
        image: Giriş görüntüsü (BGR veya gri).

    Returns:
        İyileştirilmiş gri tonlamalı görüntü.
    """
    # Gri tonlamaya çevir (eğer renkli ise)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold ile zemin-metin ayrımı
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )

    # Morfolojik temizlik - küçük gürültüleri kaldır
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned


def correct_perspective(image, contour):
    """Perspektif (trapezoid) düzeltme uygular.

    Args:
        image: Giriş görüntüsü.
        contour: Evrak köşe noktaları (4, 2).

    Returns:
        Perspektif düzeltilmiş görüntü.
    """
    return four_point_transform(image, contour)


def process_document(image_path, output_dir="output/bolum1"):
    """Tam evrak tarama pipeline'ı.

    Adımlar:
    1. Evrak konturu tespiti
    2. Maskeleme (arka plan ayrımı)
    3. Perspektif düzeltme
    4. Kontrast iyileştirme ve zemin beyazlatma

    Her adımın ara çıktısı output_dir'e kaydedilir.

    Args:
        image_path: Giriş görüntüsü dosya yolu.
        output_dir: Çıktı klasörü.

    Returns:
        İşlenmiş (kontrast iyileştirilmiş) görüntü veya hata durumunda None.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Görüntüyü oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"[HATA] Goruntu okunamadi: {image_path}")
        return None

    print(f"[BILGI] Goruntu yuklendi: {image.shape[1]}x{image.shape[0]}")
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # --- Adım 1: Evrak konturu tespiti ---
    print("[ADIM 1] Evrak konturu tespit ediliyor...")
    contour = detect_document(image)

    if contour is None:
        print("[UYARI] 4 koseli kontur bulunamadi. Goruntu oldugu gibi islenecek.")
        h, w = image.shape[:2]
        contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    else:
        print(f"  -> Evrak koselleri: {contour.astype(int).tolist()}")

    # Kontur çizilmiş görüntüyü kaydet
    contour_img = image.copy()
    pts_draw = contour.reshape(4, 1, 2).astype(np.int32)
    cv2.drawContours(contour_img, [pts_draw], -1, (0, 255, 0), 3)
    for i, p in enumerate(contour.astype(int)):
        cv2.circle(contour_img, tuple(p), 10, (0, 0, 255), -1)
        cv2.putText(contour_img, str(i), tuple(p + [15, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    path_kontur = os.path.join(output_dir, f"{basename}_1_kontur.jpg")
    cv2.imwrite(path_kontur, contour_img)
    print(f"  -> Kontur gorseli kaydedildi: {path_kontur}")

    # --- Adım 2: Maskeleme ---
    print("[ADIM 2] Maskeleme uygulaniyor...")
    masked = apply_mask(image, contour)
    path_mask = os.path.join(output_dir, f"{basename}_2_maskeli.jpg")
    cv2.imwrite(path_mask, masked)
    print(f"  -> Maskeli gorsel kaydedildi: {path_mask}")

    # --- Adım 3: Perspektif düzeltme ---
    print("[ADIM 3] Perspektif duzeltme uygulaniyor...")
    warped = correct_perspective(image, contour)
    path_persp = os.path.join(output_dir, f"{basename}_3_perspektif.jpg")
    cv2.imwrite(path_persp, warped)
    print(f"  -> Perspektif duzeltilmis gorsel kaydedildi: {path_persp}")

    # --- Adım 4: Kontrast iyileştirme ---
    print("[ADIM 4] Kontrast iyilestirme ve zemin beyazlatma...")
    enhanced = enhance_contrast(warped)
    path_kontrast = os.path.join(output_dir, f"{basename}_4_kontrast.jpg")
    cv2.imwrite(path_kontrast, enhanced)
    print(f"  -> Kontrast iyilestirilmis gorsel kaydedildi: {path_kontrast}")

    print(f"\n[TAMAM] Tum ciktilar '{output_dir}' klasorune kaydedildi.")
    return enhanced
