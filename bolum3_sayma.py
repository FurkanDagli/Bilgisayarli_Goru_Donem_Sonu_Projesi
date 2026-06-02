"""
bolum3_sayma.py - Tane Sayma
Bölüm 3: Beyaz zemin üzerinde çok taneli maddelerin sayılması.
Watershed segmentasyonu ile bitişik taneler ayrılır.
"""

import cv2
import numpy as np
import os


def preprocess(image):
    """Tane sayma için ön işleme (Güncellenmiş - Gölgelere dayanıklı).

    Gri tonlama → CLAHE (Kontrast) → Adaptive Threshold → Morfoloji.

    Args:
        image: Giriş görüntüsü (BGR).

    Returns:
        (gray_clahe, binary) tuple'ı.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Işık eşitsizliklerini gidermek için CLAHE (Kontrast İyileştirme)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Bulanıklaştırma (gürültüyü azaltmak için)
    blurred = cv2.GaussianBlur(gray_clahe, (7, 7), 0)

    # Adaptive Thresholding: Gölgelerden etkilenmeden yerel ışığa göre eşikleme
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        101, 10
    )

    # Morfolojik temizlik
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Opening: küçük gürültüleri kaldır
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing: nohutların içindeki delikleri ve kopmaları güçlü şekilde birleştir
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    return gray_clahe, binary


def segment_watershed(binary, image):
    """Watershed algoritması ile bitişik taneleri ayırır.

    Distance Transform → Yerel maksimumlar (sure foreground) →
    Watershed segmentasyonu uygular.

    Args:
        binary: İkili görüntü (taneler beyaz).
        image: Orijinal BGR görüntü (watershed için gerekli).

    Returns:
        labels: Etiket matrisi. Her tane farklı pozitif tamsayı ile etiketlenir.
                0 = arka plan, -1 = sınır.
    """
    # Kesin arka plan (sure background) - dilate ile genişlet
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    # Distance Transform - her pikselin en yakın sıfır pikseline mesafesi
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Kesin ön plan (sure foreground) - mesafe yüksek olan bölgeler
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(),
                               255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # Belirsiz bölge
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Bağlı bileşen etiketleme
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Markers'ı 1 artır (arka plan 0 yerine 1 olsun)
    markers = markers + 1

    # Belirsiz bölgeyi 0 yap (watershed'ın dolduracağı alan)
    markers[unknown == 255] = 0

    # Watershed uygula
    markers = cv2.watershed(image, markers)

    return markers


def count_and_label(image, markers):
    """Taneleri sayar ve görüntü üzerine işaretler.

    Args:
        image: Orijinal BGR görüntü.
        markers: Watershed etiket matrisi.

    Returns:
        (labeled_image, count) tuple'ı.
    """
    labeled = image.copy()
    count = 0

    unique_labels = np.unique(markers)

    for label in unique_labels:
        if label <= 1:
            continue

        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        # Dairesellik (Circularity) hesabı: 1.0 tam daire, 0'a yaklaştıkça çizgi
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Çok küçük/büyük alanları ve yuvarlak olmayanları (masadaki çizgi vb.) atla
        if area < 300 or area > 50000:
            continue
            
        if circularity < 0.35:
            continue

        count += 1

        # Konturu çiz
        color = (0, 255, 0)  # yeşil
        cv2.drawContours(labeled, [c], -1, color, 2)

        # Merkeze numara yaz
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(labeled, str(count), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                        cv2.LINE_AA)

    return labeled, count


def process_grains(image_path, output_dir="output/bolum3"):
    """Tam tane sayma pipeline'ı.

    Adımlar:
    1. Ön işleme (gri tonlama, threshold, morfoloji)
    2. Watershed segmentasyonu
    3. Sayma ve etiketleme

    Args:
        image_path: Giriş görüntüsü dosya yolu.
        output_dir: Çıktı klasörü.

    Returns:
        Toplam tane sayısı veya hata durumunda None.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Görüntüyü oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"[HATA] Goruntu okunamadi: {image_path}")
        return None

    print(f"[BILGI] Goruntu yuklendi: {image.shape[1]}x{image.shape[0]}")
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # --- Adım 1: Ön işleme ---
    print("[ADIM 1] On isleme (gri tonlama, threshold, morfoloji)...")
    gray, binary = preprocess(image)

    path_gray = os.path.join(output_dir, f"{basename}_1_gri.jpg")
    cv2.imwrite(path_gray, gray)
    print(f"  -> Gri tonlama kaydedildi: {path_gray}")

    path_binary = os.path.join(output_dir, f"{basename}_2_binary.jpg")
    cv2.imwrite(path_binary, binary)
    print(f"  -> Binary gorsel kaydedildi: {path_binary}")

    # --- Adım 2: Segmentasyon ---
    print("[ADIM 2] Watershed segmentasyonu...")
    markers = segment_watershed(binary, image)

    # Marker görselini oluştur (görselleştirme için)
    marker_vis = np.zeros_like(image)
    for label in np.unique(markers):
        if label <= 1:
            continue
        color = [int(c) for c in np.random.randint(50, 255, 3)]
        marker_vis[markers == label] = color

    path_markers = os.path.join(output_dir, f"{basename}_3_segmentasyon.jpg")
    cv2.imwrite(path_markers, marker_vis)
    print(f"  -> Segmentasyon gorseli kaydedildi: {path_markers}")

    # --- Adım 3: Sayma ve etiketleme ---
    print("[ADIM 3] Taneler sayiliyor ve etiketleniyor...")
    labeled, count = count_and_label(image, markers)

    # Sayıyı görüntünün üstüne yaz
    cv2.putText(labeled, f"Toplam: {count} adet", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    path_labeled = os.path.join(output_dir, f"{basename}_4_etiketli.jpg")
    cv2.imwrite(path_labeled, labeled)
    print(f"  -> Etiketli gorsel kaydedildi: {path_labeled}")

    print(f"\n{'=' * 40}")
    print(f"  SONUC: Toplam {count} adet tane bulundu")
    print(f"{'=' * 40}")
    print(f"[TAMAM] Tum ciktilar '{output_dir}' klasorune kaydedildi.")

    return count
