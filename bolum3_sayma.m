% bolum3_sayma.m - Taneli Madde Sayımı
% Beyaz zemin üzerinde bulunan nohut/kahve tanelerini sayar.
% İleri düzey eşikleme ve dairesellik filtresi kullanır.

pkg load image;

disp('Lutfen taneli madde (nohut vb.) goruntusunu secin...');
[dosya, yol] = uigetfile({'*.jpg;*.png;*.jpeg;*.jfif', 'Resim Dosyalari'});
if isequal(dosya, 0)
    error('Dosya secilmedi!');
end

img = imread(fullfile(yol, dosya));

% 1. Gri Tonlamaya Çevirme
gri_img = rgb2gray(img);

% Gürültüyü azaltmak için bulanıklaştırma
h_gauss = fspecial('gaussian', [7 7], 2);
gri_blur = imfilter(gri_img, h_gauss, 'replicate');

% 2. Siyah-Beyaz Formata Çevirme (Adaptive Threshold / Yerel Eşikleme)
% Octave'da adaptif threshold için arka plan aydınlığını bulup eşikliyoruz
h_bg = fspecial('average', [101 101]);
arka_plan = imfilter(gri_blur, h_bg, 'replicate');
% Zemin beyaz, nesneler koyu olduğu için koyu olanları (arka plandan 10 birim koyu) ayır
binary_img = gri_blur < (arka_plan - 10);

% 3. Boşluk Doldurma ve Gürültü Temizleme
% Nesne içindeki (parlamadan kaynaklı) siyah delikleri beyazla doldur
binary_img = imfill(binary_img, 'holes');

% Açma/Kapama ile küçük tozları temizle ve şekilleri yuvarla
se = strel('disk', 3, 0);
binary_img = imopen(binary_img, se);
binary_img = imclose(binary_img, se);

% Çok küçük parçaları haritadan sil (Area < 300 pikseller)
binary_img = bwareaopen(binary_img, 300);

% 4. Nesneleri Etiketleme ve Sayma
[labeled_img, num_objects] = bwlabel(binary_img);

% Nesnelerin özelliklerini (Alan ve Çevre) çıkar
ozellikler = regionprops(labeled_img, 'Area', 'Perimeter', 'Centroid');

gercek_tane_sayisi = 0;
gecerli_idx = [];

for i = 1:num_objects
    area = ozellikler(i).Area;
    perim = ozellikler(i).Perimeter;
    
    % Eğer çevre sıfırsa hata olmaması için atla
    if perim == 0
        continue;
    end
    
    % Dairesellik Formülü: 4 * pi * Area / Perimeter^2
    % 1'e ne kadar yakınsa o kadar daireseldir. Çizgiler 0'a yaklaşır.
    circularity = (4 * pi * area) / (perim^2);
    
    % Sadece 400'den büyük, 50000'den küçük ve dairesel (çizgi olmayan) nesneleri al
    if area > 400 && area < 50000 && circularity > 0.35
        gercek_tane_sayisi = gercek_tane_sayisi + 1;
        gecerli_idx = [gecerli_idx, i];
    end
end

% 5. Sonuçları Çizdirme ve Ekrana Yazdırma
figure(1);
imshow(img); hold on;

for k = 1:length(gecerli_idx)
    idx = gecerli_idx(k);
    merkez = ozellikler(idx).Centroid;
    
    % Merkeze numara yaz
    text(merkez(1), merkez(2), num2str(k), ...
        'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    
    % Yeşil bir nokta koy
    plot(merkez(1), merkez(2), 'g.', 'MarkerSize', 15);
end
hold off;

title(sprintf('Bulunan Tane Sayisi: %d', gercek_tane_sayisi));
fprintf('\n=================================\n');
fprintf('Gorseldeki Gercek Tane Sayisi: %d\n', gercek_tane_sayisi);
fprintf('=================================\n');
