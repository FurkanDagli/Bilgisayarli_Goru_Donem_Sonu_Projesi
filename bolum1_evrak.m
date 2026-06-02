% bolum1_evrak.m - Evrak Görüntüsünün İşlenmesi
% Bu script, uygun bir zemin üzerine konulmuş evrakın kenarlarını bulur,
% perspektifini düzeltir ve okunabilirliğini artırır.

% Image paketini yükle
pkg load image;

disp('Lutfen islenecek evrak goruntusunu secin...');
[dosya, yol] = uigetfile({'*.jpg;*.png;*.jpeg;*.jfif', 'Resim Dosyalari'});
if isequal(dosya, 0)
    error('Dosya secilmedi! Program sonlandiriliyor.');
end

% Görüntüyü oku
img = imread(fullfile(yol, dosya));
[h, w, ~] = size(img);

% 1. Gri Seviyeye Dönüştürme ve Maske Oluşturma
gri_orj = rgb2gray(img);
esik_orj = graythresh(gri_orj);
bw = im2bw(gri_orj, esik_orj);

% 2. Gürültü Temizleme ve En Büyük Alanın Bulunması
bw_temiz = imopen(bw, strel("disk", 5, 0));
[etiketler, num] = bwlabel(bw_temiz);
ozellikler = regionprops(etiketler, 'Area', 'BoundingBox');
[~, idx] = max([ozellikler.Area]);
maske = (etiketler == idx);

% 3. Sınır Noktalarının Çıkarılması
sinirlar = bwboundaries(maske);
sinir_noktalari = sinirlar{1};
y_ham = sinir_noktalari(:,1);
x_ham = sinir_noktalari(:,2);

% 4. Köşe Noktalarının Matematiksel Olarak Sıralanması
% x+y toplami -> min: sol-ust, max: sag-alt
% x-y farki   -> max: sag-ust, min: sol-alt
toplam = x_ham + y_ham;
fark = x_ham - y_ham;

[~, sol_ust_idx] = min(toplam);
[~, sag_alt_idx] = max(toplam);
[~, sag_ust_idx] = max(fark);
[~, sol_alt_idx] = min(fark);

kaynak_noktalar = [
    x_ham(sol_ust_idx), y_ham(sol_ust_idx);
    x_ham(sag_ust_idx), y_ham(sag_ust_idx);
    x_ham(sag_alt_idx), y_ham(sag_alt_idx);
    x_ham(sol_alt_idx), y_ham(sol_alt_idx)
];

% Tespit edilen köşeleri görselleştir
figure(1); imshow(img); hold on;
plot(kaynak_noktalar(:,1), kaynak_noktalar(:,2), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
line([kaynak_noktalar(:,1); kaynak_noktalar(1,1)], ...
     [kaynak_noktalar(:,2); kaynak_noktalar(1,2)], ...
     'Color', 'g', 'LineWidth', 2);
title('Otomatik Tespit Edilen Koseler');
drawnow;

% 5. Perspektif Düzeltme
% Evrakı 500x700 boyutuna oturtuyoruz
hedef_noktalar = [0, 0; 500, 0; 500, 700; 0, 700];
fprintf('Perspektif duzeltiliyor...\n');
T = maketform('projective', kaynak_noktalar, hedef_noktalar);
duzeltilmis = imtransform(img, T, 'XData', [0 500], 'YData', [0 700]);

% 6. Kontrast ve Gölge İyileştirme (Yerel Aydınlık Tabanlı)
gri_duzeltilmis = rgb2gray(duzeltilmis);
h_filtre = fspecial('average', [51 51]);
yerel_aydinlik = imfilter(gri_duzeltilmis, h_filtre, 'replicate');
% Yerel aydınlıktan biraz daha koyu olanları siyah, diğerlerini beyaz yap
final_bw = gri_duzeltilmis > (yerel_aydinlik * 0.92);

% Sonuçları göster
figure(2);
subplot(1,2,1); imshow(duzeltilmis); title('Perspektif Duzeltildi');
subplot(1,2,2); imshow(final_bw); title('Golgelerden Arindirilmis Form');

fprintf('Bolum 1: Islem Tamamlandi!\n');
