# Multi-Language Customer Support Automation

## Proje Özeti

ShopVista kurgusal e-ticaret şirketi için çok dilli (Türkçe + İngilizce) müşteri destek otomasyon sistemi. Ham şirket dokümanlarını işleyerek RAG (Retrieval-Augmented Generation) pipeline'ına uygun chunk'lara ayırır.

## Proje Yapısı

```
├── data/
│   ├── raw_docs/                  # Kaynak dokümanlar (6 adet .txt)
│   │   ├── iade_politikasi.txt    # TR — İade politikası (~400 kelime)
│   │   ├── kargo_ve_teslimat.txt  # TR — Kargo ve teslimat bilgileri
│   │   ├── urun_garantisi.txt     # TR — Garanti koşulları ve servis
│   │   ├── sss.txt                # TR+EN — Sık sorulan sorular (30 çift)
│   │   ├── shipping_policy.txt    # EN — Shipping policy
│   │   └── returns_en.txt         # EN — Returns & refunds policy
│   └── chunks/                    # İşlenmiş chunk çıktıları
│       ├── chunks.jsonl           # ✅ Production chunks (recursive, 500 token)
│       ├── chunks_recursive_256.jsonl
│       ├── chunks_recursive_500.jsonl
│       ├── chunks_recursive_1000.jsonl
│       ├── chunks_sentence_500.jsonl
│       ├── chunks_fixed_500.jsonl
│       └── experiment_summary.json
├── src/
│   └── document_processor.py      # Doküman işleme ve chunking pipeline
├── requirements.txt               # Bağımlılıklar (yalnızca stdlib)
├── .gitignore                     # Git ignore kuralları
└── README.md
```

## Bağımlılıklar ve Kurulum

### Sistem Gereksinimleri
- **Python 3.10+** (3.11.3 ile test edilmiştir)
- Hiçbir harici Python paketi gerekmez — yalnızca standart kütüphane kullanılmaktadır

### Kurulum

1. **Hazırlık** (opsiyonel — sanal ortam önerilir):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. **Bağımlılıklar** (yalnızca Python versiyonu kontrol etmek isterseniz):
   ```bash
   # Bağımlılık dosyası bilgilendirme amaçlı; gerçek kurulum gerekmez
   cat requirements.txt
   ```

### Dosya Yapısı
- `requirements.txt` — Proje bağımlılıkları ve Python versiyonu specifications
- `.gitignore` — Git'in ignore edeceği dosya/klasörler (cache, .venv, IDE dosyaları, vs.)

## Çalıştırma

```bash
python src/document_processor.py
```

Çıktı `data/chunks/` klasörüne kaydedilir.

---

## Teknik Detaylar: Pipeline İşleyişi

### 1. Doküman Yükleme & Temizleme
- UTF-8 (BOM dahil) ve Latin-1 fallback ile encoding sorunları çözülür
- Unicode NFC normalizasyonu uygulanır
- Non-breaking space, fazla boşluk, 3+ ardışık newline temizlenir
- Her satır strip edilir

### 2. Dil Tespiti
- Dosya adına göre birincil dil atanır (`shipping_policy.txt` → en, `iade_politikasi.txt` → tr)
- `sss.txt` gibi çok dilli dosyalarda chunk bazında dil tespiti yapılır (Türkçe özel karakterler vs. İngilizce stop word sayısı)

### 3. Chunking
Her chunk'a şu metadata eklenir:
```json
{
  "text": "...",
  "source": "iade_politikasi.txt",
  "language": "tr",
  "chunk_id": 3,
  "chunk_method": "recursive",
  "chunk_size_setting": 500,
  "token_count": 487
}
```

### 4. Çıktı
- `.jsonl` formatında — satır başına bir JSON objesi
- Production çıktısı: `data/chunks/chunks.jsonl`

---

## Chunking Deneyleri ve Analiz

### A. Chunk Size Karşılaştırması (Recursive metod, overlap=50)

| Chunk Size | Chunk Sayısı | Ort. Token | Min | Max | Karakteristik |
|---|---|---|---|---|---|
| **256 token** | 26 chunk | 197.5 | 107 | 252 | Granüler, kısa bağlam penceresi. Her chunk tek konu/alt-bölüm kapsar. Retrieval precision yüksek ama recall düşebilir çünkü cevap birden fazla chunk'a dağılır. |
| **500 token** | 12 chunk | 369.5 | 112 | 483 | Dengeli. Bir bölümün tamamını veya büyük kısmını tek chunk'ta tutar. RAG için ideal — hem yeterli bağlam hem de odaklı içerik. |
| **1000 token** | 7 chunk | 597.7 | 396 | 963 | Geniş bağlam. Birden fazla bölüm tek chunk'a düşer. Retrieval recall yüksek ama noise artar, LLM'in ilgisiz bilgiyi filtrelemesi gerekir. |

**Seçim: 500 token**

Gerekçe:
- Dokümanlarımız orta uzunlukta (4–10 KB), bölüm bazlı yapıya sahip
- 500 token, bir bölümün (~200-600 kelime) tamamını veya büyük kısmını tek chunk'ta tutabiliyor
- 256 token'da iade koşulları gibi listeler chunk sınırında bölünüyor; 1000 token'da ise "kargo ücretleri" ile "teslimat sorunları" gibi farklı konular aynı chunk'ta birleşiyor
- 500 token, retrieval precision ve recall arasında en iyi dengeyi sağlıyor
- Overlap (50 token) sayesinde bölüm sınırlarındaki bilgi kaybı minimize ediliyor

### B. Chunking Metodu Karşılaştırması (chunk_size=500, overlap=50)

| Metod | Chunk Sayısı | Ort. Token | Min | Max | Yaklaşım |
|---|---|---|---|---|---|
| **Recursive Character** | 12 | 369.5 | 112 | 483 | Bölüm başlıkları → paragraf → satır → cümle → kelime sırasıyla böler |
| **Sentence-based** | 12 | 369.5 | 96 | 497 | Cümle sınırlarında böler, sonra cümleleri token limitine göre gruplar |
| **Fixed-size** | 13 | 342.3 | 16 | 500 | Sabit token penceresiyle mekanik bölme |

**Detaylı Karşılaştırma:**

| Metod | Avantaj | Dezavantaj |
|---|---|---|
| **Recursive Character** | Doküman yapısını korur, anlamsal bütünlük yüksek. Başlık sınırlarında bölmeyi tercih eder. | Uygulama karmaşıklığı biraz daha fazla |
| **Sentence-based** | Cümle ortasında kopma olmaz | Bölüm/paragraf yapısını görmezden gelir. Listeleri ve tabloları iyi parçalayamaz |
| **Fixed-size** | En basit uygulama, tahmin edilebilir chunk boyutu | Anlam ortasında kesiyor, başlıkları gövdeden ayırıyor, min=16 token'lık artık chunk üretiyor, retrieval kalitesi en düşük |

**Seçim: Recursive Character Text Splitter**

Gerekçe:
- Dokümanlarımız numaralı bölümlerle (1. İADE SÜRESİ, 2. RETURN CONDITIONS vb.) net yapıya sahip
- Recursive splitter bu yapıyı tanıyor ve önce bölüm sınırlarında bölmeyi deniyor
- "İade adımları" gibi sıralı listeler tek chunk'ta kalıyor (sentence splitter bunları parçalayabiliyor)
- Fixed-size splitter testlerinde, "6. HASARLI VEYA EKSİK ÜRÜNLER" başlığının bir önceki bölümün sonuyla aynı chunk'a düşmesi gibi semantik hatalar gözlemlendi
- Recursive yöntem, tüm deney konfigürasyonları arasında en tutarlı anlamsal chunk'ları üretiyor

---

## Tutarlılık Kuralları (Ground Truth)

Tüm dokümanlar aşağıdaki bilgilerde birebir tutarlıdır:

| Bilgi | Değer |
|---|---|
| İade süresi | 14 takvim günü / 14 calendar days |
| Ücretsiz kargo (yurtiçi) | 500 TL ve üzeri |
| Ücretsiz kargo (uluslararası) | 75 USD ve üzeri |
| Kargo ücreti (yurtiçi) | 29,90 TL (500 TL altı) |
| Kargo ücreti (uluslararası) | 15 USD (75 USD altı) |
| Yurtiçi teslimat | 2–5 iş günü |
| Uluslararası teslimat | 7–15 iş günü |
| Garanti (elektronik) | 2 yıl |
| Garanti (diğer) | 1 yıl |
| Müşteri hizmetleri | 0850 123 45 67 |
| E-posta | destek@shopvista.com.tr / support@shopvista.com.tr |
