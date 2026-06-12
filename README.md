
# Is-This-Text-Ai-
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GuardinTheDev/Is-This-Text-Ai-/blob/main/Is-This-This-Text-AI.ipynb)

# 🤖 Yapay Zeka Metin Detektörü (AI Text Detector)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GuardinTheDev/Is-This-Text-Ai-/blob/main/Is-This-This-Text-AI.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)

Bu proje, verilen bir metnin **yapay zeka tarafından mı** yoksa **gerçek bir insan tarafından mı** yazıldığını tespit etmek amacıyla geliştirilmiş bir Derin Öğrenme modelidir. DistilBERT mimarisi üzerine fine-tune edilmiş olan model, Gradio arayüzü sayesinde kolayca test edilebilir.

> **Not:** Model, 50–100 kelimelik metinlerde en yüksek doğruluk oranını sağlamaktadır.

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Donanım ve Yazılım Gereksinimleri](#-donanım-ve-yazılım-gereksinimleri)
- [Klasör Yapısı](#-klasör-yapısı)
- [Kullanılan Veri Setleri](#-kullanılan-veri-setleri)
- [Kullanılan Kütüphaneler](#-kullanılan-kütüphaneler)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
  - [Yöntem 1: Google Colab (Önerilen)](#yöntem-1-google-colab-önerilen)
  - [Yöntem 2: Yerel Kurulum](#yöntem-2-yerel-kurulum)
- [Örnek Veriler](#-örnek-veriler)
- [Model Hakkında](#-model-hakkında)
- [Veritabanı Bilgisi](#-veritabanı-bilgisi)
- [Sık Sorulan Sorular](#-sık-sorulan-sorular)

---

## ✨ Özellikler

- **DistilBERT** tabanlı ikili sınıflandırma (AI / İnsan)
- **Gradio** arayüzü ile kolay test imkânı
- Eğitilmiş model ağırlıkları Google Drive'dan otomatik indirilir — tekrar eğitim gerekmez
- Sıcaklık ayarı (temperature scaling) ile kalibre edilmiş olasılık çıktıları
- GPU (CUDA) desteği; GPU yoksa CPU ile otomatik çalışır

---

## 💻 Donanım ve Yazılım Gereksinimleri

### Yazılım
| Gereksinim | Sürüm |
|---|---|
| Python | 3.8 veya üzeri |
| CUDA (isteğe bağlı) | 11.8+ |
| Google Colab | Ücretsiz T4 GPU yeterlidir |

### Donanım (Yerel Kurulum)
| Senaryo | Minimum | Önerilen |
|---|---|---|
| Sadece tahmin (inference) | 4 GB RAM, CPU | 8 GB RAM, herhangi bir GPU |
| Model yeniden eğitimi | 16 GB RAM, GPU (8 GB VRAM) | 32 GB RAM, T4 / V100 GPU |

> ⚠️ **Google Colab kullanıyorsanız** ekstra donanım gerekmez. Ücretsiz T4 GPU, hem eğitim hem tahmin için yeterlidir.

---


## 💻 Donanım ve Yazılım Gereksinimleri

### Yazılım
| Gereksinim | Sürüm |
|---|---|
| Python | 3.8 veya üzeri |
| CUDA (isteğe bağlı) | 11.8+ |
| Google Colab | Ücretsiz T4 GPU yeterlidir |


## 📂 Klasör Yapısı

```
Is-This-Text-Ai-/
│
├── Is-This-This-Text-AI.ipynb   # Ana model: veri hazırlama, eğitim, Gradio arayüzü
├── Tokenizer.ipynb               # Yardımcı notebook: tokenizasyon ve metin metrikleri
│                                 # (Perplexity, Burstiness, Flesch-Kincaid hesaplama)
├── newDataset                    # Veri seti referans dosyası
├── requirements.txt              # Python bağımlılıkları
├── .env.example                  # Ortam değişkenleri şablonu
├── sample_data/
│   ├── ai_samples.txt            # Örnek yapay zeka metinleri
│   └── human_samples.txt         # Örnek insan metinleri
└── README.md                     # Bu dosya
```

---

## 📊 Kullanılan Veri Setleri

Model, aşırı öğrenmeyi (overfitting) ve veri yanlılığını (dataset bias) önlemek amacıyla iki farklı kaynaktan gelen veriler harmanlanarak eğitilmiştir:

### 1. Kaggle DAIGT Veri Seti
- **Kaynak:** [Yunij/kaggle-comp-daigt (Hugging Face)](https://huggingface.co/datasets/Yunij/kaggle-comp-daigt)
- **İçerik:** Yapay zeka ve insan tarafından yazılmış genel metinler
- **Kullanım amacı:** Ana eğitim verisi

### 2. Wikitext-103 (Wikipedia) Veri Seti
- **Kaynak:** [Salesforce/wikitext — wikitext-103-raw-v1 (Hugging Face)](https://huggingface.co/datasets/Salesforce/wikitext)
- **İçerik:** Wikipedia'dan alınmış uzun ve akademik İngilizce makaleler
- **Kullanım amacı:** Modelin resmi/akademik üslubu yapay zeka sanmasını önlemek için 10.000 adet insan yazısı (etiket: 0) olarak eklendi

### Veri Bölümleme
| Set | Oran | Açıklama |
|---|---|---|
| Eğitim (train) | %80 | Modelin öğrendiği veriler |
| Doğrulama (validation) | %20 | Model başarısının ölçüldüğü veriler |

---

## 📦 Kullanılan Kütüphaneler

| Kütüphane | Sürüm | Kullanım Amacı |
|---|---|---|
| `transformers` | ≥4.30 | DistilBERT modelini yüklemek, eğitmek ve tokenizer işlemleri için |
| `datasets` | ≥2.12 | Hugging Face Hub'dan veri setlerini çekmek, filtrelemek ve birleştirmek için |
| `torch` (PyTorch) | ≥2.0 | Tüm tensör hesaplamaları ve GPU hızlandırması için temel derin öğrenme iskeleti |
| `scikit-learn` | ≥1.2 | Accuracy ve F1-Score metriklerini hesaplamak için |
| `gradio` | ≥3.40 | Modeli test edebilmek için görsel web arayüzü (UI) oluşturmak için |
| `gdown` | ≥4.7 | Eğitilmiş model dosyalarını Google Drive'dan otomatik indirmek için |
| `numpy` | ≥1.24 | Nümerik hesaplamalar için |
| `statistics` | Standart kütüphane | Burstiness (cümle uzunluğu sapması) hesabı için |
| `re` | Standart kütüphane | Metin temizleme ve cümle ayrıştırma için |

### Tokenizer.ipynb'e Özgü Kütüphaneler
| Kütüphane | Kullanım Amacı |
|---|---|
| `torch` + `transformers` | GPT-2 modeliyle Perplexity (şaşkınlık skoru) hesabı |
| `statistics` | Burstiness (standart sapma) hesabı |
| `math`, `re`, `collections` | Flesch-Kincaid okunabilirlik skoru ve token frekansı analizi |

---

## 🚀 Kurulum ve Çalıştırma

### Yöntem 1: Google Colab (Önerilen)

Bu yöntem, yerel kurulum gerektirmez. Tek tıkla çalışır.

**Adım 1 — Colab'ı Aç**

Sayfanın en üstündeki [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GuardinTheDev/Is-This-Text-Ai-/blob/main/Is-This-This-Text-AI.ipynb) butonuna tıkla.

**Adım 2 — GPU'yu Etkinleştir**

```
Üst Menü → Çalışma Zamanı (Runtime) → Çalışma Zamanı Türünü Değiştir
→ Donanım Hızlandırıcısı: T4 GPU → Kaydet
```

> GPU olmadan da çalışır ama tahmin süresi daha uzun olur.

**Adım 3 — Tümünü Çalıştır**

```
Üst Menü → Çalışma Zamanı → Tümünü Çalıştır (Run All)
```

Kod otomatik olarak şunları yapacaktır:
1. Gerekli kütüphaneleri yükler
2. Eğitilmiş model ağırlıklarını Google Drive'dan indirir
3. Modeli hafızaya yükler
4. Son hücrede bir **Gradio arayüzü** başlatır (tıklanabilir link üretir)

**Adım 4 — Test Et**

Gradio arayüzünde çıkan linke tıkla, metin kutusuna analiz etmek istediğin metni yapıştır ve **"Analiz Et"** butonuna bas.

---

### Yöntem 2: Yerel Kurulum

#### Gereksinimler
- Python 3.8+
- pip
- (İsteğe bağlı) CUDA destekli GPU

#### Adım 1 — Repoyu Klonla

```bash
git clone https://github.com/GuardinTheDev/Is-This-Text-Ai-.git
cd Is-This-Text-Ai-
```

#### Adım 2 — Sanal Ortam Oluştur (Önerilen)

```bash
# venv ile
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# veya conda ile
conda create -n ai-detector python=3.10
conda activate ai-detector
```

#### Adım 3 — Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

#### Adım 4 — Ortam Değişkenlerini Ayarla

```bash
cp .env.example .env
# .env dosyasını bir metin editörü ile aç ve gerekli değerleri doldur
```

#### Adım 5 — Notebook'u Çalıştır

```bash
jupyter notebook Is-This-This-Text-AI.ipynb
```

veya doğrudan script olarak çalıştırmak istersen:

```bash
jupyter nbconvert --to script Is-This-This-Text-AI.ipynb
python Is-This-This-Text-AI.py
```

> **Model Yeniden Eğitmek İstersen:** Notebook içindeki `YENIDEN_EGIT = False` satırını `True` yaparak modeli baştan eğitebilirsin. Bu işlem GPU ile ~30–45 dakika, CPU ile birkaç saat sürebilir.

---

## 🧪 Örnek Veriler

### Yapay Zeka Tarafından Yazılmış Metin Örneği
```
Furthermore, the integration of renewable energy sources into the existing power grid requires a comprehensive restructuring of distributional logistics.
Recent advancements in smart grid technologies offer a viable framework for mitigating systemic volatility; however, the initial infrastructural expenditures
 remain restrictively high. Consequently, small-scale municipalities often lack the fiscal resilience necessary to transition away from traditional fossil fuels.
This technological disparity ultimately exacerbates existing regional socioeconomic inequalities
```
**Beklenen Sonuç:** Yapay Zeka Tarafından Yazılmış 🤖 (~%75–85 YZ olasılığı)

---

### İnsan Tarafından Yazılmış Metin Örneği
```
Last night I had coffee with an old friend. It had been so long since we'd seen each other, I couldn't even remember exactly how many years had passed.
 When we sat in the cafe, there was an awkward silence at first, but then everything flowed as usual. That's how it is with some people;
 even after years, the connection doesn't disappear.
```
**Beklenen Sonuç:** İnsan Tarafından Yazılmış 👨‍💻 (~%70–80 İnsan olasılığı)

---

## 🧠 Model Hakkında

### Mimari
- **Temel Model:** `distilbert-base-uncased` (Hugging Face)
- **Görev:** İkili sınıflandırma (0 = İnsan, 1 = Yapay Zeka)
- **Maksimum Token:** 512

### Eğitim Parametreleri
| Parametre | Değer |
|---|---|
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epoch | 3 |
| Weight Decay | 0.01 |
| Optimizer | AdamW (Trainer varsayılanı) |

### Karar Mekanizması
Model çıktısı, iki hiperparametre ile kalibre edilir:

| Parametre | Varsayılan Değer | Açıklama |
|---|---|---|
| `AI_ESIK` | 70.0 | Modelin "YZ" kararı verebilmesi için gereken minimum YZ olasılığı (%) |
| `SICAKLIK` | 3.5 | Olasılık dağılımını yumuşatır; yüksek değer daha dengeli yüzdeler üretir |

### Eğitilmiş Model Dosyası
Model dosyaları (~260 MB) Google Drive'da barındırılmaktadır. Notebook çalıştırıldığında `gdown` kütüphanesi aracılığıyla otomatik olarak `/content/en_iyi_detektor_modeli/` klasörüne indirilir.

Model klasörü şu dosyaları içerir:
```
en_iyi_detektor_modeli/
├── config.json
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── model.safetensors   # veya pytorch_model.bin
```

---

## 🗄️ Veritabanı Bilgisi

Bu projede **geleneksel bir ilişkisel veritabanı kullanılmamaktadır.** Veri yönetimi tamamen Hugging Face ekosistemi üzerinden yürütülür:

| Katman | Teknoloji | Açıklama |
|---|---|---|
| Ham veri | Hugging Face Datasets | `load_dataset()` ile buluttan RAM'e çekilir |
| İşlenmiş veri | Bellek içi (in-memory) | `concatenate_datasets()` ile birleştirilir |
| Model ağırlıkları | `.safetensors` / `.bin` | `trainer.save_model()` ile diske kaydedilir |
| Model depolama | Google Drive | `gdown` ile otomatik indirilir |

Veritabanı kurulum adımı gerekmez.

---

## ❓ Sık Sorulan Sorular

**S: Model Türkçe metinlerde de çalışır mı?**
A: Model İngilizce (`distilbert-base-uncased`) eğitilmiştir. Türkçe metinlerde sonuçlar güvenilir olmayabilir. Türkçe için `dbmdz/bert-base-turkish-cased` gibi bir model tercih edilmelidir.

**S: `gdown` indirme başarısız olursa ne yapmalıyım?**
A: Notebook içindeki `YENIDEN_EGIT = False` satırını `True` yaparak modeli baştan eğitebilirsiniz.

**S: Gradio linki neden açılmıyor?**
A: Colab ücretsiz hesaplarda Gradio `share=True` linki birkaç saate kadar aktif kalır. Notebook'u yeniden çalıştırarak yeni bir link elde edebilirsiniz.

---


