"""
main.py - Görüntü İşleme Ödevi CLI Arayüzü
Ana giriş noktası. Tüm bölümleri komut satırından çalıştırır.
"""

import argparse
import sys
import os


def run_bolum1(args):
    """Bölüm 1: Evrak Tarama"""
    from bolum1_evrak import process_document

    if not args.input:
        print("[HATA] --input parametresi gerekli (goruntu dosya yolu).")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[HATA] Dosya bulunamadi: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  BOLUM 1: EVRAK TARAMA")
    print("=" * 60)
    process_document(args.input, args.output or "output/bolum1")


def run_bolum2(args):
    """Bölüm 2: Optik Form Okuyucu"""
    from bolum2_optik import batch_grade, grade_paper, load_answer_key_from_file

    if not args.input:
        print("[HATA] --input parametresi gerekli (form klasoru veya dosya yolu).")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[HATA] Yol bulunamadi: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  BOLUM 2: OPTIK FORM OKUYUCU")
    print("=" * 60)

    if os.path.isdir(args.input):
        # Toplu değerlendirme
        batch_grade(args.input, answer_key_path=args.answer_key)
    else:
        # Tek form değerlendirme
        if not args.answer_key:
            print("[HATA] Tek form icin --answer-key parametresi gerekli.")
            sys.exit(1)
        answer_key = load_answer_key_from_file(args.answer_key)
        result = grade_paper(args.input, answer_key)
        if result:
            print(f"\n  Ogrenci No : {result['student_id']}")
            print(f"  Dogru      : {result['dogru']}")
            print(f"  Yanlis     : {result['yanlis']}")
            print(f"  Bos        : {result['bos']}")
            print(f"  Puan       : %{result['puan']}")
            print()
            print("  Detaylar:")
            for d in result['detay']:
                print(d)


def run_bolum3(args):
    """Bölüm 3: Tane Sayma"""
    from bolum3_sayma import process_grains

    if not args.input:
        print("[HATA] --input parametresi gerekli (goruntu dosya yolu).")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[HATA] Dosya bulunamadi: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  BOLUM 3: TANE SAYMA")
    print("=" * 60)
    process_grains(args.input, args.output or "output/bolum3")


def run_generate_omr(args):
    """Optik form test seti üret"""
    from generate_omr import generate_test_set

    n = args.count or 10
    output = args.output or "test_images/bolum2"

    print("=" * 60)
    print("  OPTIK FORM URETICI")
    print("=" * 60)
    generate_test_set(output_dir=output, n_students=n)


def main():
    parser = argparse.ArgumentParser(
        description="Goruntu Isleme Odevi - CLI Arayuzu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornek Kullanim:
  # Bolum 1: Evrak tarama
  python main.py --bolum 1 --input foto.jpg

  # Bolum 2: Optik form okuma (klasor)
  python main.py --bolum 2 --input test_images/bolum2/

  # Bolum 2: Tek form okuma
  python main.py --bolum 2 --input form.png --answer-key cevaplar.txt

  # Bolum 3: Tane sayma
  python main.py --bolum 3 --input taneler.jpg

  # Optik form test seti uret
  python main.py --generate-omr --count 10
        """
    )

    parser.add_argument("--bolum", type=int, choices=[1, 2, 3],
                        help="Calistirilacak bolum numarasi (1, 2 veya 3)")
    parser.add_argument("--input", "-i", type=str,
                        help="Giris dosya/klasor yolu")
    parser.add_argument("--output", "-o", type=str,
                        help="Cikti klasoru (varsayilan: output/bolumX)")
    parser.add_argument("--answer-key", "-k", type=str,
                        help="Cevap anahtari dosya yolu (Bolum 2 icin)")
    parser.add_argument("--generate-omr", action="store_true",
                        help="Optik form test seti uret")
    parser.add_argument("--count", "-n", type=int, default=10,
                        help="Uretilecek ogrenci formu sayisi (varsayilan: 10)")

    args = parser.parse_args()

    if args.generate_omr:
        run_generate_omr(args)
    elif args.bolum == 1:
        run_bolum1(args)
    elif args.bolum == 2:
        run_bolum2(args)
    elif args.bolum == 3:
        run_bolum3(args)
    else:
        parser.print_help()
        print("\n[BILGI] Lutfen --bolum veya --generate-omr parametresi belirtin.")


if __name__ == "__main__":
    main()
