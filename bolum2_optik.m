% bolum2_optik.m - Optik Form Okuyucu
% Daha önce ürettiğimiz test_images/bolum2 içindeki öğrenci formlarını okur.

pkg load image;

% 1. Form ve Cevap Anahtarı Ayarları
PAGE_W = 800;
PAGE_H = 1100;
MARKER_SIZE = 25;
% Orijinal python kodundaki cevap anahtarı: (0=A, 1=B, 2=C, 3=D, 4=E)
% Lütfen 'test_images/bolum2/beklenen_sonuclar.txt' dosyasındaki gercek cevaplarla karsilastirin.
% Biz burada örnek bir cevap anahtari dizisi tanimliyoruz.
cevap_anahtari = [1, 3, 0, 4, 2, 1, 1, 4, 0, 2, 3, 0, 2, 4, 1, 3, 2, 4, 0, 1]; 
SIKLAR = {'A', 'B', 'C', 'D', 'E'};

disp('Lutfen okunacak optik formu (ogrenci_01.png vb.) secin...');
[dosya, yol] = uigetfile({'*.png;*.jpg', 'Resim Dosyalari'});
if isequal(dosya, 0)
    error('Dosya secilmedi!');
end

img = imread(fullfile(yol, dosya));
if size(img, 3) == 3
    gri = rgb2gray(img);
else
    gri = img;
end

% 2. Form Hizalama (Marker bulma ve Perspektif Düzeltme)
% Formun köşe noktalarini (siyah kareleri) bulup 800x1100'e oturtalim
bw_inv = gri < 100; % Siyah yerler 1 oldu
[etiketler, num] = bwlabel(bw_inv);
ozellikler = regionprops(etiketler, 'Area', 'Centroid', 'BoundingBox');

merkezler = [];
for i = 1:num
    alan = ozellikler(i).Area;
    bb = ozellikler(i).BoundingBox;
    en = bb(3); boy = bb(4);
    aspect_ratio = en / boy;
    
    % Marker boyutları yaklaşik 25x25 (Alan ~625). Tolerans: 200-2000
    if alan > 200 && alan < 2000 && aspect_ratio > 0.6 && aspect_ratio < 1.4
        merkezler = [merkezler; ozellikler(i).Centroid];
    end
end

if size(merkezler, 1) < 4
    warning('4 adet kose isareti bulunamadi, dogrudan 800x1100 yapiliyor...');
    hizali_form = imresize(gri, [PAGE_H, PAGE_W]);
else
    % Sol-ust, sag-ust, sol-alt, sag-alt ayriştirmasi
    toplam = merkezler(:,1) + merkezler(:,2);
    fark = merkezler(:,1) - merkezler(:,2);
    [~, tl_idx] = min(toplam);
    [~, br_idx] = max(toplam);
    [~, tr_idx] = max(fark);
    [~, bl_idx] = min(fark);
    
    kaynak_noktalar = [
        merkezler(tl_idx, :);
        merkezler(tr_idx, :);
        merkezler(br_idx, :);
        merkezler(bl_idx, :)
    ];
    
    % Python'daki MARKER_CENTERS karsiligi (x, y)
    hedef_noktalar = [
        42, 42;
        757, 42;
        757, 1057;
        42, 1057
    ];
    
    T = maketform('projective', kaynak_noktalar, hedef_noktalar);
    hizali_form = imtransform(gri, T, 'XData', [1 PAGE_W], 'YData', [1 PAGE_H]);
end

figure(1); imshow(hizali_form); title('Hizalanmis Optik Form');

% 3. Cevaplari Okuma (Python'daki Sabit Koordinatlarla)
fprintf('\n--- Optik Form Okunuyor ---\n');

% Soru okuma ayarlari
col_origins = [150, 555; 500, 555]; % x, y
h_spacing = 45;
v_spacing = 36;
radius = 10;
FILL_THRESHOLD = 180; % Gribin bu degerden kucukse (koyuysa) doludur

okunan_cevaplar = zeros(1, 20);

for sutun_idx = 1:2
    ox = col_origins(sutun_idx, 1);
    oy = col_origins(sutun_idx, 2);
    
    for soru = 1:10
        soru_no = (sutun_idx - 1) * 10 + soru;
        cy = oy + (soru - 1) * v_spacing;
        
        sik_degerleri = zeros(1, 5);
        for sik = 1:5
            cx = ox + (sik - 1) * h_spacing;
            
            % Baloncugu kes (x ve y ekseninde 10'ar piksel geri/ileri)
            kutu = hizali_form(cy-radius : cy+radius, cx-radius : cx+radius);
            sik_degerleri(sik) = mean(kutu(:)); % Ortalama parlaklik
        end
        
        % En koyu şıkkı bul
        [min_val, min_idx] = min(sik_degerleri);
        
        if min_val < FILL_THRESHOLD
            okunan_cevaplar(soru_no) = min_idx; % 1=A, 2=B, 3=C...
        else
            okunan_cevaplar(soru_no) = 0; % Boş
        end
    end
end

% 4. Değerlendirme
dogru = 0; yanlis = 0; bos = 0;
for i = 1:20
    gercek_cevap = cevap_anahtari(i) + 1; % Python 0'dan, Matlab 1'den baslar
    okunan = okunan_cevaplar(i);
    
    if okunan == 0
        bos = bos + 1;
        fprintf('Soru %2d: BOS\n', i);
    elseif okunan == gercek_cevap
        dogru = dogru + 1;
        fprintf('Soru %2d: %s (DOGRU)\n', i, SIKLAR{okunan});
    else
        yanlis = yanlis + 1;
        fprintf('Soru %2d: %s (YANLIS - Cevap: %s)\n', i, SIKLAR{okunan}, SIKLAR{gercek_cevap});
    end
end

puan = (dogru / 20) * 100;
fprintf('\n========================\n');
fprintf('DOGRU  : %d\n', dogru);
fprintf('YANLIS : %d\n', yanlis);
fprintf('BOS    : %d\n', bos);
fprintf('PUAN   : %% %.1f\n', puan);
fprintf('========================\n');
