# Is-This-Text-Ai-
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GuardinTheDev/Is-This-Text-Ai-/blob/main/Is-This-This-Text-AI.ipynb)

#  Yapay Zeka Metin Detektörü (AI Text Detector)

Bu proje, verilen bir metnin yapay zeka tarafından mı yoksa gerçek bir insan tarafından mı yazıldığını tespit etmek amacıyla geliştirilmiş bir Derin Öğrenme (Deep Learning) modelidir. Projede `DistilBERT` mimarisi kullanılmış olup, modelin ezber yapmasını (dataset bias) önlemek amacıyla farklı kaynaklardan gelen veriler harmanlanarak eğitilmiştir.
---

## 📂 Kullanılan Veri Setleri

Modelin eğitimi ve doğrulaması (validation) için aşağıdaki veri setleri harmanlanarak kullanılmıştır:

Kaggle DAIGT Veri Seti: Yapay zeka ve insan tarafından yazılmış genel metinleri içerir.
Bağlantı: [Yunij/kaggle-comp-daigt (Hugging Face)](https://huggingface.co/datasets/Yunij/kaggle-comp-daigt)

Wikitext (Wikipedia) Veri Seti: Modelin resmi, akademik ve ciddi dili yapay zeka sanmasını engellemek amacıyla, dengeleyici (gerçek insan metni) olarak projeye entegre edilmiştir. Kütüphane güncellemeleri ve veri hacmini artırmak adına Salesforce/wikitext veri setinin wikitext-103-raw-v1 sürümü kullanılarak sisteme 10.000 adet uzun ve formel insan makalesi enjekte edilmiştir.
Bağlantı: https://huggingface.co/datasets/Salesforce/wikitext

---

## 🛠️ Kullanılan Kütüphaneler ve Amaçları

Projenin geliştirilmesinde aşağıdaki Python kütüphanelerinden faydalanılmıştır:

* **`transformers`**: Hugging Face'in sağladığı bu kütüphane, `DistilBERT` modelini yüklemek, eğitmek (Trainer) ve metinleri modelin anlayacağı token'lara çevirmek (Tokenizer) için kullanılmıştır.
* **`datasets`**: Dev boyuttaki veri setlerini internetten (Hugging Face Hub) hızlıca çekmek, filtrelemek ve birleştirmek için kullanılmıştır.
* **`torch` (PyTorch)**: Modelin arka plandaki tüm karmaşık tensör (matris) hesaplamalarını ve GPU (Ekran Kartı) hızlandırmasını sağlayan temel derin öğrenme iskeletidir.
* **`scikit-learn`**: Modelin eğitim sonrasındaki başarı oranını (Accuracy) ve F1-Score değerlerini matematiksel olarak hesaplamak için kullanılmıştır.
* **`gradio`**: Geliştirilen modelin terminal ekranından çıkarılıp, son kullanıcının rahatça test edebileceği butonlu ve görsel bir web arayüzüne (UI) dönüştürülmesi için kullanılmıştır.
* **`gdown`**: Eğitilmiş devasa model dosyalarını Google Drive üzerinden yerel ortama otomatik indirmek için kullanılmıştır.

---

## 🗄️ Veritabanı ve Model Yönetim Süreci

Bu projede geleneksel bir ilişkisel veritabanı (MySQL, PostgreSQL vb.) yerine, **Büyük Dil Modeli (LLM) ağırlıkları ve Hugging Face Dataset mimarisi** kullanılmıştır. Veri ve model yönetimi şu basamaklarla gerçekleşmektedir:

1. **Verilerin Çekilmesi:** `datasets` kütüphanesi ile veriler buluttan RAM'e anlık olarak çekilir ve harmanlanır.
2. **Model Ağırlıklarının İndirilmesi:** Proje her çalıştığında modelin baştan eğitilmemesi için, daha önce eğitilmiş model ağırlıkları `.bin` veya `.safetensors` formatında tutulur. Kod içindeki `gdown` scripti sayesinde bu ağırlıklar klasöre (`/content/en_iyi_detektor_modeli`) otomatik olarak indirilir.
3. **Kayıt:** Eğer eğitim baştan yapılırsa, `trainer.save_model()` fonksiyonu ile yeni ağırlıklar yerel dizine veritabanı objesi gibi kaydedilir.

---

## ⚙️ Kurulum ve Çalıştırma Adımları

Projeyi kendi bilgisayarınızda veya Google Colab üzerinde çalıştırmak için aşağıdaki adımları izleyiniz:

**Adım 1:** Projeyi kendi bilgisayarınıza indirip kurmanıza gerek kalmadan, doğrudan tarayıcı üzerinden hızlıca test edebilmeniz için Google Colab altyapısı tercih edilmiştir.
**Projeyi Colab'da Açın:** En yukarıdaki butona tıklayarak projenin kodlarını doğrudan Google Colab üzerinde açabilirsiniz.

**Adım 2: GPU'yu Aktif Edin (Önemli)** Eğer model eğtilmek isteniyorsa "Model Eğitim" bloğunda "YENIDEN_EGIT = False" kısmını "True" yapaarak modeli eğitebilirsiniz.Modelin hızlı çalışması için Colab açıldığında üst menüden **Çalışma Zamanı (Runtime) > Çalışma Zamanı Türünü Değiştir (Change runtime type)** seçeneğine tıklayın. Donanım Hızlandırıcıyı **T4 GPU** olarak seçip kaydedin.

Not:Model 50-100 kelimelik kelime bandında daha yüksek isabet oranı sağlamaktadır


**Adım 3: Programı Başlatın** * Üst menüden **Çalışma Zamanı > Tümünü Çalıştır (Run all)** seçeneğine tıklayın.
* Eğer Google Drive bağlantısı için izin isterse onaylayın.
* Kod otomatik olarak kütüphaneleri kuracak, model dosyalarını indirecek ve son hücrede test yapabileceğiniz tıklanabilir bir **Gradio Web Arayüzü (Live Link)** oluşturacaktır.
