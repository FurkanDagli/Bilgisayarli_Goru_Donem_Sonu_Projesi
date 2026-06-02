"""
utils.py - Ortak yardımcı fonksiyonlar
Görüntü işleme ödevi için paylaşılan yardımcı araçlar.
"""

import cv2
import numpy as np


def resize_image(image, width=800):
    """Görüntüyü verilen genişliğe orantılı olarak yeniden boyutlandırır.

    Args:
        image: Giriş görüntüsü (BGR).
        width: Hedef genişlik (piksel).

    Returns:
        Yeniden boyutlandırılmış görüntü.
    """
    h, w = image.shape[:2]
    if w == width:
        return image.copy()
    ratio = width / float(w)
    new_height = int(h * ratio)
    resized = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def order_points(pts):
    """4 noktayı sol-üst, sağ-üst, sağ-alt, sol-alt sırasına koyar.

    Args:
        pts: (4, 2) şeklinde numpy dizisi.

    Returns:
        Sıralı (4, 2) float32 numpy dizisi.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype="float32")

    # Toplam: sol-üst en küçük, sağ-alt en büyük
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # sol-üst
    rect[2] = pts[np.argmax(s)]  # sağ-alt

    # Fark: sağ-üst en küçük fark, sol-alt en büyük fark
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # sağ-üst
    rect[3] = pts[np.argmax(diff)]  # sol-alt

    return rect


def four_point_transform(image, pts):
    """4 noktalı perspektif dönüşümü uygular.

    Args:
        image: Giriş görüntüsü.
        pts: 4 köşe noktası (4, 2) numpy dizisi.

    Returns:
        Perspektif düzeltilmiş görüntü.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Yeni görüntü genişliği
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Yeni görüntü yüksekliği
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Hedef noktalar
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Dönüşüm matrisi ve uygulama
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
