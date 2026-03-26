# Multi-Language Customer Support Automation

## Proje Özeti

ShopVista kurgusal e-ticaret şirketi için çok dilli (Türkçe + İngilizce) müşteri destek otomasyon sistemi. Ham şirket dokümanlarını işleyip chunk'lara ayırır, multilingual embedding ile vektör veritabanına yükler ve RAG (Retrieval-Augmented Generation) pipeline'ı üzerinden doğal dilde müşteri sorularını yanıtlar.

## Proje Yapısı

```
├── data/
│   ├── raw_docs/                  # Kaynak dokümanlar (6 adet .txt)
│   │   ├── iade_politikasi.txt    # TR — İade politikası
│   │   ├── kargo_ve_teslimat.txt  # TR — Kargo ve teslimat bilgileri
│   │   ├── urun_garantisi.txt     # TR — Garanti koşulları ve servis
│   │   ├── sss.txt                # TR+EN — Sık sorulan sorular (30 çift)
│   │   ├── shipping_policy.txt    # EN — Shipping policy
│   │   └── returns_en.txt         # EN — Returns & refunds policy
│   ├── chunks/                    # İşlenmiş chunk çıktıları
│   │   ├── chunks.jsonl           # ✅ Production chunks (recursive, 500 token)
│   │   ├── chunks_recursive_*.jsonl
│   │   ├── chunks_sentence_500.jsonl
│   │   ├── chunks_fixed_500.jsonl
│   │   └── experiment_summary.json
│   └── vectordb/                  # ChromaDB persist directory (gitignore'd)
├── src/
│   ├── document_processor.py      # Doküman işleme ve chunking pipeline
│   ├── vectorstore.py             # Embedding ve ChromaDB vektör veritabanı
│   ├── rag_pipeline.py            # RAG pipeline (retrieval + Gemini LLM)
│   ├── conversation_manager.py    # Konuşma geçmişi bellek yönetimi
│   ├── api_key_rotator.py         # Google API 429 hataları için Round-Robin rotasyon
│   ├── eval_llm_wrapper.py        # Ragas için LangChain Gemini LLM sarmalayıcısı
│   └── eval_embed_wrapper.py      # Ragas için LangChain HuggingFace sarmalayıcısı
├── evaluation/
│   ├── test_set.json              # Ragas için 20 soruluk değerlendirme seti
│   └── eval_results.csv           # Pipeline test sonuçları metrikleri
├── notebooks/                     
│   └── evaluation.ipynb           # Performans (Ragas) ve multi-turn test defteri
├── .env                           # API anahtarları(github üzerinde paylaşılmıyor,    kullanıcı oluşturmalı)
├── requirements.txt
├── .gitignore
└── README.md
```

## Kurulum


### Sistem Gereksinimleri
- **Python 3.10+** (3.11.3 ile test edilmiştir)

### Adımlar

1. **Sanal ortam oluşturun:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **`.env` dosyası oluşturun** (proje kök dizininde):
   ```
   GEMINI_API_KEY=your_google_gemini_api_key
   # Çoklu api hesapları varsa, Round-Robin olarak dönmek için sırayla eklenebilir:
   GEMINI_API_KEY_2=your_google_gemini_api_key_2
   GEMINI_API_KEY_3=your_google_gemini_api_key_3
   HF_API_KEY=your_huggingface_api_key
   ```

## Çalıştırma

### 1. Doküman İşleme & Chunking
```bash
python src/document_processor.py
```
Kaynak dokümanları işler, chunk'lara ayırır → `data/chunks/chunks.jsonl`

### 2. Embedding & Vektör Veritabanı
```bash
python src/vectorstore.py
```
Chunk'ları embed eder, ChromaDB'ye yükler → `data/vectordb/`

### 3. RAG Pipeline
```bash
python src/rag_pipeline.py
```
Örnek sorularla RAG pipeline'ını çalıştırır (dil tespiti → retrieval → LLM cevap üretimi).

### 4. Değerlendirme (Evaluation) & Multi-turn Testleri
Jupyter Notebook ortamını aktif edin:
```bash
jupyter notebook notebooks/evaluation.ipynb
```
Bu notebook:
1. `evaluation/test_set.json` içerisindeki değerlendirme setini alır, RAG pipeline'a sorar.
2. `ragas` kütüphanesi kullanarak **Faithfulness** ve **AnswerRelevancy** skorlarını çıkarıp `.csv` kaydeder.
3. Multi-Turn konuşma simülasyonunu farklı dillerdeki senaryolarla (Türkçe/İngilizce, zamir takibi ile) çalıştırır.

---

## Teknik Detaylar

### Doküman İşleme (`document_processor.py`)
- **Encoding:** UTF-8 (BOM dahil) ve Latin-1 fallback
- **Temizleme:** Unicode NFC normalizasyonu, non-breaking space, fazla boşluk/newline temizliği
- **Dil tespiti:** Dosya adına göre birincil dil; çok dilli dosyalarda (`sss.txt`) chunk bazında tespit
- **Chunking:** Recursive character splitting (bölüm başlıkları → paragraf → cümle → kelime)
- **Çıktı:** `.jsonl` formatı — her satır bir JSON chunk objesi

Chunk metadata yapısı:
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

### Embedding & Vektör Veritabanı (`vectorstore.py`)
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ dil)
- **Veritabanı:** ChromaDB (persistent, cosine distance)
- **Collection:** `customer_support_chunks` — 12 production chunk
- Cross-lingual retrieval testi: TR ve EN sorgular arasında %80 overlap (excellent alignment)

### Konuşma Geçmişi / Multi-Turn Sözlük Optimizasyonu (`conversation_manager.py`)
- Müşterinin ardışık sorular sormasını ve (Örn: "Peki bu cihazda nasıl olur?") gibi zamirlere ("bu", "şu") göre asistanın konuyu algılamasını sağlar.
- `deque` tabanlı FIFO buffer tekniği kullanarak, token patlamasını önlemek maksadıyla sadece son N (örn. 5 adet) mesaj history içine yansıtılır.
- Her mesajın içeriğine `Kullanıcı: ...` ve `Asistan: ...` formatları append edilip sistem promptunda `"conversation_history"` olarak kullanılır.

### RAG Pipeline (`rag_pipeline.py`)
- **Akış:** Soru → Dil tespiti (`langdetect`) → Önceki session hafızasını okuma → Embedding + ChromaDB top-3 retrieval → Geçmiş sohbet bağlamı ile beraber prompt oluşturma → LLM → Cevap
- **API Hata Toleransı & Rate Limit:** 
  - `api_key_rotator.py`: Env'deki birden fazla anahtarı Round-Robin değiştirir.
  - "Exponential Backoff": Google API'dan (429 - Quota Excedeed) cevabı dönerse süreyi katlayarak yeniden dener.
- **LLM:** Google Gemini 2.5 Flash (`google-genai` SDK)
- **System prompt kuralları:**
  - Sadece verilen context'e dayanarak cevap ver
  - Cevabın dilini sorunun diliyle eşleştir
  - Bilgi yoksa müşteri hizmetlerine yönlendir
  - Cevabın sonunda kaynak dokümanları belirt

### Ragas İle Performans Testleri
- Sistem kalitesini ölçmek adına RAGAS (v0.4.3) tercih edildi. Langchain sarmalayıcıları olarak Pydantic temelli bağımsız yapılar yazıldı (`eval_llm_wrapper.py`, `eval_embed_wrapper.py`). 
- **Veri Seti:** Karışık, Edge-case ve Dokümanlarda bulunmayan zor sorular içeren bir `test_set.json` (20 Soru) manuel yaratıldı.
- RAGAS'ın *Asenkron API Rate Limit* hatalarını önlemek adına `RunConfig(max_workers=1)` kurularak limitasyon senkronize sağlandı.

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

## Kullanılan Teknolojiler

| Bileşen | Teknoloji |
|---|---|
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) |
| Vektör DB | ChromaDB (persistent, cosine distance) |
| LLM | Google Gemini 2.5 Flash |
| Dil tespiti | langdetect |
| API key yönetimi | python-dotenv (`.env` dosyasından) |

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
