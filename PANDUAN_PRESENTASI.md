# 📊 Panduan Presentasi: Website Segmentasi Pelanggan Berbasis Machine Learning

> **Proyek:** Customer Segmentation Dashboard  
> **Algoritma:** K-Means Clustering  
> **Mata Kuliah:** Machine Learning — Semester 6  
> **Tanggal Demo:** 6 Juli 2026

---

## 📌 Daftar Isi

1. [Gambaran Umum Website](#1-gambaran-umum-website)
2. [Tujuan & Manfaat](#2-tujuan--manfaat)
3. [Arsitektur Sistem](#3-arsitektur-sistem)
4. [Fitur-Fitur Website](#4-fitur-fitur-website)
5. [Cara Kerja Algoritma](#5-cara-kerja-algoritma)
6. [4 Segmen Pelanggan yang Dideteksi](#6-4-segmen-pelanggan-yang-dideteksi)
7. [Langkah-Langkah Demo (Skrip Presentasi)](#7-langkah-langkah-demo-skrip-presentasi)
8. [Pertanyaan yang Mungkin Muncul (Q&A)](#8-pertanyaan-yang-mungkin-muncul-qa)

---

## 1. Gambaran Umum Website

Website ini adalah **sistem analitik interaktif** yang menggunakan algoritma **K-Means Clustering** untuk mengklasifikasikan pelanggan marketplace ke dalam **4 segmen perilaku** yang berbeda. Cukup dengan mengisi 10 data profil pelanggan, sistem langsung memberikan hasil segmentasi beserta karakteristik dan rekomendasi strategi pemasaran yang sesuai.

| Aspek | Detail |
|---|---|
| **Nama Sistem** | Customer Segmentation Dashboard |
| **Bahasa** | Python (Backend Flask) + HTML/CSS/JavaScript (Frontend) |
| **Model ML** | K-Means Clustering **(4 Cluster)** |
| **Model File** | `model_kmeans_final.pkl` (dilatih dengan scikit-learn) |
| **Normalisasi Data** | Z-Score Standardization |
| **Jumlah Fitur Input** | 10 fitur |
| **Jumlah Segmen** | **4 cluster** |

---

## 2. Tujuan & Manfaat

### 🎯 Tujuan
- Membantu bisnis/marketplace dalam **memahami karakteristik perilaku pelanggan** secara otomatis.
- Mengklasifikasikan pelanggan ke dalam kelompok-kelompok yang memiliki kesamaan pola belanja.
- Memberikan **rekomendasi strategi pemasaran** yang tepat sasaran untuk setiap segmen.

### 💡 Manfaat

#### Untuk Bisnis / Marketplace
| Manfaat | Penjelasan |
|---|---|
| **Personalisasi Pemasaran** | Setiap segmen mendapat strategi promosi yang berbeda dan relevan |
| **Efisiensi Anggaran** | Tidak perlu membuang biaya iklan ke pelanggan yang salah segmen |
| **Peningkatan Konversi** | Tawaran yang tepat sasaran meningkatkan kemungkinan pembelian |
| **Retensi Pelanggan** | Pelanggan premium dapat diprioritaskan dengan layanan eksklusif |
| **Pengambilan Keputusan Data-Driven** | Keputusan bisnis berdasarkan pola data nyata, bukan asumsi |

#### Untuk Akademik / Machine Learning
| Manfaat | Penjelasan |
|---|---|
| **Implementasi Nyata K-Means** | Demonstrasi penerapan algoritma unsupervised learning di dunia nyata |
| **Pipeline ML Lengkap** | Mulai dari preprocessing (Z-Score) hingga deployment sebagai web app |
| **Interpretabilitas Model** | Setiap cluster memiliki deskripsi dan karakteristik yang mudah dipahami |

---

## 3. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                    BROWSER (Frontend)                   │
│  index.html — Form input 10 fitur pelanggan             │
│  + Quick-Fill buttons + Hasil segmentasi dinamis        │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST / JavaScript
                         ▼
┌─────────────────────────────────────────────────────────┐
│               BACKEND — Flask (web-app.py)              │
│  1. Terima data form dari pengguna                      │
│  2. Lakukan Z-Score Standardization manual              │
│  3. Prediksi cluster dengan model K-Means (.pkl)        │
│  4. Kembalikan info segmen ke template HTML             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            MODEL — model_kmeans_final.pkl               │
│  K-Means 4 Cluster, 10 Dimensi Fitur                   │
│  Dilatih dengan scikit-learn & joblib                   │
└─────────────────────────────────────────────────────────┘
```

> **Catatan Penting:** Website ini juga dilengkapi dengan **implementasi K-Means langsung di JavaScript** (client-side). Artinya prediksi dapat berjalan **secara instan di browser** tanpa perlu server, menggunakan centroid yang sama persis dengan model Python.

---

## 4. Fitur-Fitur Website

### 4.1 Form Input 10 Fitur Pelanggan

Website meminta pengguna mengisi 10 data profil pelanggan berikut:

| No | Fitur | Tipe Input | Rentang Data |
|---|---|---|---|
| 1 | **Jenis Kelamin** | Dropdown | Laki-laki / Perempuan |
| 2 | **Usia** | Angka | 18 – 70 tahun |
| 3 | **Pendapatan Tahunan** | Angka | 15 – 140 Juta Rp |
| 4 | **Skor Belanja** | Angka | 1 – 100 |
| 5 | **Metode Pembayaran Favorit** | Dropdown | COD / Dompet Digital / Kartu Kredit / Transfer Bank |
| 6 | **Jumlah Transaksi per Bulan** | Angka | 1 – 10 kali |
| 7 | **Rata-rata Nilai Transaksi** | Desimal | 0.5 – 6.5 Juta Rp |
| 8 | **Jumlah Kategori Produk Dibeli** | Dropdown | 1 / 2 / 3 kategori |
| 9 | **Frekuensi Penggunaan Promo** | Dropdown | 0 – 6 kali |
| 10 | **Rata-rata Rating Diberikan** | Desimal | 2.0 – 5.0 |

### 4.2 Quick-Fill (Isi Cepat Contoh Data)

Tombol pintar untuk mengisi form secara otomatis dengan data representatif tiap segmen:
- 👴 **Senior Hemat** — data pelanggan senior yang berhati-hati
- 💎 **Premium Loyal** — data pelanggan berpenghasilan tinggi dan loyal
- 🛍️ **Promo Hunter** — data pelanggan muda yang suka diskon
- 💼 **Kaya Pasif** — data pelanggan kaya tapi jarang belanja

> **Fungsi:** Sangat berguna saat demo untuk menunjukkan berbagai hasil segmentasi tanpa harus mengisi form manual berulang kali.

### 4.3 Tombol Segmentasi & Hasil Real-time

- Klik tombol **"🚀 Jalankan Segmentasi Pelanggan"**
- Sistem memproses data dan menampilkan hasil dengan **animasi slide-up** yang halus
- Hasil mencakup:
  - 🏷️ **Badge Cluster** (nomor cluster hasil prediksi)
  - 🔖 **Nama Segmen** (nama deskriptif yang mudah dipahami)
  - 📝 **Profil Segmen** (deskripsi perilaku pelanggan)
  - 📋 **Karakteristik Utama** (4 poin kunci dengan ikon centang ✓)
  - 💡 **Rekomendasi Pemasaran** (strategi bisnis yang disarankan)

### 4.4 Desain Responsif

- Tampilan optimal di **desktop, tablet, dan smartphone**
- Grid 2 kolom yang otomatis berubah ke 1 kolom di layar kecil

---

## 5. Cara Kerja Algoritma

### Alur Proses Segmentasi (Step-by-Step)

```
INPUT (10 fitur mentah dari pengguna)
    │
    ▼
STEP 1: Z-SCORE STANDARDIZATION
    │  Formula: z = (nilai - rata_rata_training) / std_deviasi_training
    │  Tujuan: Menyamakan skala semua fitur agar tidak ada yang mendominasi
    │
    ▼
STEP 2: MENGHITUNG JARAK EUCLIDEAN
    │  Hitung jarak vektor z terhadap 5 centroid cluster
    │  Formula: distance = √Σ(z_i - centroid_i)²
    │
    ▼
STEP 3: ASSIGN CLUSTER
    │  Pilih cluster dengan jarak Euclidean terkecil
    │  → Pelanggan dimasukkan ke cluster tersebut
    │
    ▼
OUTPUT: Nama segmen, karakteristik, & rekomendasi bisnis
```

### Mengapa Z-Score Standardization?

Data setiap fitur memiliki skala yang sangat berbeda:
- Usia: `18–70` vs Pendapatan: `15.000.000–140.000.000`

Tanpa normalisasi, K-Means akan lebih "bias" ke fitur dengan angka besar. Z-Score mengubah semua fitur ke skala standar (mean=0, std=1) sehingga setiap fitur berkontribusi secara setara.

### Statistik Normalisasi (dari Data Training)

| Fitur | Mean (μ) | Std Dev (σ) |
|---|---|---|
| Jenis Kelamin | 0.56 | 0.496 |
| Usia | 38.85 | 13.969 |
| Pendapatan Tahunan (Juta Rp) | 60.56 | 26.264 |
| Skor Belanja | 50.20 | 25.823 |
| Transaksi per Bulan | 2.62 | 1.423 |
| Rata-rata Nilai Transaksi (Juta Rp) | 2.478 | 1.136 |
| Jumlah Kategori Produk | 1.235 | 0.490 |
| Frekuensi Promo | 2.075 | 1.610 |
| Rata-rata Rating | 3.566 | 0.534 |
| Metode Pembayaran | 1.055 | 1.002 |

---

## 6. 4 Segmen Pelanggan yang Dideteksi (Sesuai Model Final)

> Semua angka di bawah diambil langsung dari **nilai centroid `model_kmeans_final.pkl`** yang telah diverifikasi. Hasil ini konsisten dengan grafik distribusi segmen pada notebook analisis.

### 📊 Tabel Ringkasan Semua Cluster

| Fitur | C0 🛍️ Promo Hunter | C1 👴 Senior Hemat | C2 💎 Premium Loyal | C3 💼 Kaya Pasif |
|---|---|---|---|---|
| **Usia** | ~25 th | ~50 th | ~31 th | ~40 th |
| **Pendapatan** | ~26 Jt | ~43 Jt | ~78 Jt | ~82 Jt |
| **Skor Belanja** | ~75 ⬆️ | ~39 ⬇️ | ~73 ⬆️ | ~25 ⬇️ |
| **Transaksi/Bulan** | ~3.6x | ~2x | ~3.7x | ~1.4x ⬇️ |
| **Nilai/Transaksi** | ~1.2 Jt ⬇️ | ~1.7 Jt | ~3.2 Jt ⬆️ | ~3.4 Jt ⬆️ |
| **Kategori Produk** | ~1.4 | ~1.1 | ~1.1 | ~1.0 |
| **Frek. Promo** | ~4x ⬆️ | ~3x | ~0.9x ⬇️ | ~1x ⬇️ |
| **Rating** | ~3.77 | ~3.57 | ~3.72 | ~3.20 ⬇️ |

> 📊 **Distribusi Pelanggan (dari Notebook):** Senior Hemat ±62 pelanggan • Premium Loyal ±48 • Promo Hunter ±46 • Kaya Pasif ±44

---

### 🛍️ Cluster 0 — Pelanggan Muda Promo Hunter
| Aspek | Detail |
|---|---|
| **Profil** | Pelanggan muda, pendapatan rendah, sangat aktif berbelanja terutama saat ada promo/diskon |
| **Usia Rata-rata** | **~25 tahun** *(centroid: 25.19)* |
| **Pendapatan** | **~26 Juta Rp/tahun** *(centroid: 26.23)* |
| **Skor Belanja** | **~75 (sangat tinggi)** *(centroid: 74.61)* |
| **Transaksi/Bulan** | **~3-4 kali** *(centroid: 3.55)* |
| **Nilai Transaksi** | **~1.2 Juta Rp** per transaksi *(centroid: 1.21)* — kecil namun frekuensi tinggi |
| **Frekuensi Promo** | **~4 kali/bulan** *(centroid: 4.05)* — tertinggi di semua cluster |
| **Rating** | ~3.77/5 |
| **Rekomendasi** | Flash sale, cashback e-wallet, voucher gratis ongkir minimum belanja rendah, produk trendi harga terjangkau |

---

### 👴 Cluster 1 — Pelanggan Senior Hemat
| Aspek | Detail |
|---|---|
| **Profil** | Pelanggan paruh baya hingga senior, pendapatan menengah ke bawah, berbelanja hati-hati dan terencana |
| **Usia Rata-rata** | **~50 tahun** *(centroid: 49.93)* |
| **Pendapatan** | **~43 Juta Rp/tahun** *(centroid: 42.91)* |
| **Skor Belanja** | **~39 (rendah)** *(centroid: 38.99)* |
| **Transaksi/Bulan** | **~2 kali** *(centroid: 1.98)* |
| **Nilai Transaksi** | **~1.69 Juta Rp** per transaksi *(centroid: 1.69)* |
| **Frekuensi Promo** | **~3 kali/bulan** *(centroid: 3.02)* — cukup aktif menggunakan promo |
| **Rating** | ~3.57/5 |
| **Rekomendasi** | Paket bundling kebutuhan pokok/kesehatan, diskon khusus lansia, antarmuka aplikasi yang simpel dan mudah dinavigasi |

---

### 💎 Cluster 2 — Pelanggan Premium Loyal
| Aspek | Detail |
|---|---|
| **Profil** | Pelanggan usia produktif dengan pendapatan tinggi, daya beli kuat, sangat aktif, loyal, dan tidak peduli promo |
| **Usia Rata-rata** | **~31 tahun** *(centroid: 30.71)* |
| **Pendapatan** | **~78 Juta Rp/tahun** *(centroid: 78.43)* |
| **Skor Belanja** | **~73 (sangat tinggi)** *(centroid: 72.71)* |
| **Transaksi/Bulan** | **~3-4 kali** *(centroid: 3.73)* |
| **Nilai Transaksi** | **~3.2 Juta Rp** per transaksi *(centroid: 3.23)* |
| **Frekuensi Promo** | **~1 kali/bulan** *(centroid: 0.88)* — paling jarang menggunakan promo |
| **Rating** | ~3.72/5 |
| **Rekomendasi** | Program VIP Club eksklusif, layanan prioritas, gratis ongkir instan, early access koleksi terbaru, penawaran personal berkualitas tinggi |

---

### 💼 Cluster 3 — Pelanggan Kaya Pasif
| Aspek | Detail |
|---|---|
| **Profil** | Pendapatan tinggi namun jarang berbelanja; skor belanja rendah dan engagement rendah terhadap marketplace |
| **Usia Rata-rata** | **~40 tahun** *(centroid: 40.37)* |
| **Pendapatan** | **~82 Juta Rp/tahun** *(centroid: 82.26)* — tertinggi kedua |
| **Skor Belanja** | **~25 (sangat rendah)** *(centroid: 25.06)* |
| **Transaksi/Bulan** | **~1.4 kali** *(centroid: 1.36)* — paling jarang di semua cluster |
| **Nilai Transaksi** | **~3.35 Juta Rp** per transaksi *(centroid: 3.35)* — besar meski jarang |
| **Frekuensi Promo** | **~1 kali/bulan** *(centroid: 0.96)* — jarang menggunakan promo |
| **Rating** | ~3.20/5 — terendah, menunjukkan kurang kepuasan |
| **Rekomendasi** | Personal marketing email/notifikasi produk premium/luxury, kampanye re-engagement, promo eksklusif akhir pekan untuk memancing kembali minat belanja |

---

## 7. Langkah-Langkah Demo (Skrip Presentasi)

### 📋 Urutan Presentasi yang Disarankan

#### **Pembukaan (2 menit)**
> *"Selamat pagi/siang. Saya akan mendemonstrasikan website Customer Segmentation Dashboard yang dibangun menggunakan algoritma Machine Learning K-Means Clustering. Website ini bertujuan untuk membantu bisnis marketplace dalam mengidentifikasi tipe pelanggan mereka secara otomatis berdasarkan 10 data perilaku belanja."*

---

#### **Langkah 1 — Buka Website (30 detik)**
1. Jalankan backend: `python web-app.py` di terminal (port 5001)
2. Buka browser → akses `http://localhost:5001`
3. Tunjukkan tampilan dashboard secara keseluruhan
4. Jelaskan bahwa ini adalah sistem berbasis Flask (Python) + HTML/JS

> 💬 *"Ini adalah tampilan utama website. Di bagian atas terdapat form input data pelanggan, dan di bawahnya nanti akan muncul hasil segmentasi secara langsung."*

---

#### **Langkah 2 — Jelaskan Input Form (2 menit)**
1. Arahkan ke form input dengan 10 kolom
2. Jelaskan setiap fitur:
   - **Demografis:** Jenis kelamin, usia, pendapatan tahunan
   - **Perilaku Belanja:** Skor belanja, frekuensi transaksi, nilai transaksi rata-rata
   - **Preferensi:** Kategori produk, penggunaan promo, rating, metode pembayaran
3. Tunjukkan validasi input (contoh: usia hanya 18–70)

> 💬 *"Sistem meminta 10 fitur yang mencerminkan profil lengkap seorang pelanggan — mulai dari data demografis hingga kebiasaan belanja mereka di marketplace."*

---

#### **Langkah 3 — Demo Quick-Fill (1 menit)**
1. Klik tombol **"🛍️ Promo Hunter"**
2. Tunjukkan bahwa form terisi otomatis dan ada efek highlight biru
3. Jelaskan fungsi tombol Quick-Fill sebagai data contoh representatif

> 💬 *"Untuk mempermudah demo, saya sediakan tombol 'Isi Cepat' yang otomatis mengisi form dengan data representatif masing-masing segmen."*

---

#### **Langkah 4 — Jalankan Segmentasi Pertama (2 menit)**
1. Klik **"🚀 Jalankan Segmentasi Pelanggan"**
2. Tunjukkan animasi hasil yang muncul (slide-up effect)
3. Jelaskan komponen hasil:
   - Badge cluster (nomor cluster)
   - Nama segmen dan ikon
   - Profil/deskripsi segmen
   - 4 karakteristik utama
   - Rekomendasi pemasaran

> 💬 *"Hasilnya muncul langsung dengan detail lengkap: nama segmen, karakteristik profil pelanggan, dan yang terpenting — rekomendasi strategi pemasaran yang bisa langsung diterapkan oleh tim marketing."*

---

#### **Langkah 5 — Demo Segmen Lain (2 menit)**
1. Klik **"💎 Premium Loyal"** → jalankan segmentasi → tunjukkan hasilnya berbeda
2. Klik **"💼 Kaya Pasif"** → jalankan → tunjukkan perbedaan cluster
3. Bandingkan hasil dari 2 pelanggan berbeda secara visual

> 💬 *"Perhatikan bahwa meski keduanya berpendapatan tinggi, Cluster 2 (Premium Loyal) dan Cluster 3 (Kaya Pasif) memiliki rekomendasi yang berbeda karena pola belanja mereka berbeda. Inilah keunggulan segmentasi."*

---

#### **Langkah 6 — Jelaskan Cara Kerja Algoritma (3 menit)**
1. Jelaskan pipeline singkat: **Input → Z-Score → Euclidean Distance → Assign Cluster**
2. Tunjukkan bahwa model disimpan dalam file `model_kmeans_final.pkl`
3. Jelaskan bahwa prediksi di browser menggunakan **centroid yang sama** dari model Python

> 💬 *"Di balik layar, sistem bekerja dalam 3 tahap: pertama, data dinormalisasi dengan Z-Score agar skala setiap fitur setara. Kedua, dihitung jarak Euclidean ke 5 centroid cluster. Ketiga, pelanggan dimasukkan ke cluster dengan jarak terdekat."*

---

#### **Langkah 7 — Demo Input Manual (2 menit)** *(Opsional)*
1. Isi form secara manual dengan data fiktif
2. Contoh: Perempuan, 28 tahun, pendapatan 30 juta, skor belanja 80, sering promo
3. Jalankan segmentasi dan baca hasilnya

> 💬 *"Sekarang saya coba input data manual. Pelanggan muda, berpendapatan rendah, skor belanja tinggi, dan sering pakai promo — mari kita lihat sistem akan menempatkan dia di segmen mana."*

---

#### **Penutup (1 menit)**
> *"Kesimpulannya, website ini mengimplementasikan Machine Learning K-Means secara end-to-end — dari training model Python hingga deployment sebagai web application yang interaktif. Sistem ini bisa langsung digunakan oleh tim bisnis untuk mengoptimalkan strategi pemasaran berdasarkan profil nyata pelanggan mereka."*

---

## 8. Pertanyaan yang Mungkin Muncul (Q&A)

### ❓ "Mengapa memilih K-Means?"
> K-Means adalah algoritma clustering yang efisien dan hasilnya mudah diinterpretasikan. Karena data pelanggan tidak memiliki label (unlabeled), K-Means sebagai metode unsupervised learning sangat cocok digunakan. Kompleksitasnya O(n·k·i) membuatnya cepat bahkan untuk dataset besar.

---

### ❓ "Bagaimana menentukan jumlah cluster = 4?"
> Jumlah cluster ditentukan menggunakan metode **Elbow Method** — yaitu dengan memplot nilai inertia (SSE) terhadap berbagai jumlah cluster, lalu memilih titik "siku" di mana penambahan cluster tidak lagi signifikan menurunkan inertia. Nilai **k=4** menghasilkan 4 segmen yang bermakna secara bisnis dan terbukti dari distribusi data di notebook analisis.

---

### ❓ "Seberapa akurat model ini?"
> K-Means bukan masalah klasifikasi, jadi tidak ada "akurasi" dalam arti konvensional. Kualitas model diukur dengan **Silhouette Score** dan **inertia (SSE)**. Model yang baik menghasilkan cluster yang kohesif (dalam cluster rapat) dan terpisah jelas antar cluster.

---

### ❓ "Apakah bisa digunakan untuk data pelanggan baru secara real-time?"
> Ya, itulah yang didemonstrasikan di sini. Setelah model dilatih dan centroid tersimpan, prediksi untuk pelanggan baru bisa dilakukan secara instan — bahkan langsung di browser tanpa server menggunakan JavaScript.

---

### ❓ "Apa perbedaan frontend JavaScript vs backend Flask untuk prediksi?"
> Keduanya menggunakan logika yang identik (Z-Score + Euclidean Distance + nearest centroid). Versi JavaScript berjalan di browser (lebih cepat, tidak butuh server). Versi Flask menggunakan model `.pkl` asli dari scikit-learn sebagai validasi dan backup.

---

### ❓ "Bagaimana jika ada fitur baru yang ingin ditambahkan?"
> Model perlu di-retrain ulang dengan fitur baru tersebut. Statistik normalisasi (mean, std) juga harus diupdate. Centroid baru kemudian bisa dieksport dan diperbarui di kode JavaScript.

---

## ⚡ Tips Presentasi

- ✅ Pastikan server Flask sudah berjalan sebelum presentasi dimulai (`python web-app.py`)
- ✅ Siapkan browser dengan tab sudah terbuka di `http://localhost:5001`
- ✅ Gunakan fitur **Quick-Fill** untuk demo yang cepat dan mulus
- ✅ Demo minimal **3 segmen berbeda** untuk menunjukkan variasi output
- ✅ Fokus pada **relevansi bisnis** dari setiap rekomendasi yang muncul
- ⚠️ Pastikan port 5001 tidak digunakan aplikasi lain
- ⚠️ Nonaktifkan notifikasi browser agar tidak mengganggu presentasi

---

## ⚠️ Catatan Penting: Ketidaksesuaian Web App vs Notebook

Saat ini terdapat perbedaan antara hasil notebook dan konfigurasi web app:

| Aspek | Notebook (Hasil Analisis) | Web App (index.html + web-app.py) |
|---|---|---|
| **Jumlah Segmen** | **4 segmen** | **5 segmen** (ada Cluster 4 “Aktif Eksploratif”) |
| **Quick-Fill Buttons** | — | Ada tombol "Aktif Eksploratif" yang seharusnya dihapus |

**Penyebab:** Model `.pkl` yang tersimpan dilatih dengan `n_clusters=5`, sementara analisis di notebook menggunakan `k=4`.

**Solusi yang Disarankan (pilih salah satu):**
1. 🔄 **Retrain model** dengan `k=4` → simpan ulang sebagai `model_kmeans_final.pkl` → update centroid di `web-app.py` dan `index.html`
2. ✏️ **Update web app** saja → hapus Cluster 4 dari `cluster_info` dan `centers` di kedua file, serta hapus tombol Quick-Fill "Aktif Eksploratif"

---

*📄 Dokumen ini dibuat sebagai panduan demo website Customer Segmentation Dashboard berbasis K-Means Clustering.*
