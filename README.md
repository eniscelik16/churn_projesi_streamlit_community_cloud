# 🏦 Bank Customer Churn Prediction Dashboard

Bu proje, banka müşterilerinin bankayı terk etme (churn) riskini makine öğrenmesi algoritmaları kullanarak tahmin eden ve yöneticilere stratejik kararlar almaları için etkileşimli bir arayüz sunan profesyonel bir makine öğrenmesi web uygulamasıdır.

## 🌐 Canlı Demo

Uygulamayı bilgisayarınıza kurmadan, doğrudan tarayıcınız üzerinden test edebilirsiniz:

🔗 **[Uygulamaya Gitmek İçin Tıklayın](https://churn-simulator.streamlit.app/)**

**Demo Giriş Bilgileri:**
* **Kullanıcı Adı:** `admin`
* **Şifre:** `123`

---

## 🚀 Öne Çıkan Özellikler

* 📊 **Executive Overview:** Müşteri demografik verileri ve banka genelindeki terk (churn) oranlarının görselleştirildiği detaylı yönetim paneli. (Light ve Dark mod ile tam uyumlu dinamik UI tasarımı).
* 👤 **Single Customer Analysis:** Tekil müşteri verileri girilerek anlık terk riski hesaplama, beklenen finansal kayıp (CLTV tabanlı) analizi ve **SHAP (SHapley Additive exPlanations)** ile riski artıran/azaltan faktörlerin şeffaf analizi.
* 🤖 **Otonom AI Agent:** Müşterinin risk seviyesi ve bakiyesine göre anlık pazarlama stratejileri (VIP elde tutma, otomatik SMS kampanyası, çapraz satış) üreten kural tabanlı akıllı asistan.
* 🧪 **What-If Simulator:** Yöneticilere "Müşteriye kredi kartı verirsek veya bakiyesi artarsa terk etme ihtimali ne kadar düşer?" sorusunun cevabını veren interaktif simülatör.
* 📂 **Batch Analysis:** Toplu müşteri verilerinin yer aldığı CSV dosyalarını okuyarak finansal kayıp riskine göre müşterileri önceliklendiren ve raporlayan modül.

---

## 🛠️ Kullanılan Teknolojiler

* **Arayüz & Görselleştirme:** Streamlit, Plotly
* **Makine Öğrenmesi & AI:** Scikit-Learn, XGBoost, Random Forest
* **Açıklanabilir Yapay Zeka (XAI):** SHAP
* **Veri İşleme:** Pandas, NumPy

---

## 💻 Kendi Bilgisayarınızda Çalıştırma Adımları

Bu projeyi GitHub'dan indirip kendi bilgisayarınızda (lokalinizde) test etmek isterseniz sırasıyla şu adımları izlemelisiniz:

**Adım 1: Gerekli Kütüphaneleri Yükleyin**

Projenin ihtiyaç duyduğu eklentileri bilgisayarınıza kurmak için terminalinize şu komutu yazın:
```bash
pip install -r requirements.txt
```
**Adım 2: Yönetici Şifrelerini Ayarlayın**

Güvenlik standartları gereği şifre dosyası GitHub'a yüklenmemiştir. Uygulamanın giriş (Login) ekranını geçebilmek için proje ana klasörünüzün içine `.streamlit` adında yeni bir gizli klasör açın. İçine `secrets.toml` adında bir dosya oluşturup şu kodları yapıştırın:
```toml
[auth]
username = "admin"
password = "123"
```
**Adım 3: Uygulamayı Başlatın**

Tüm kurulumlar tamamlandı. Terminalinize aşağıdaki komutu yazarak uygulamayı tarayıcınızda açabilirsiniz:
```bash
streamlit run app.py
