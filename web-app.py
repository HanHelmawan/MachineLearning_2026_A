from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 1. Load Model K-Means Final
model_kmeans = joblib.load('model_kmeans_final.pkl')

# 2. Statistik Deskriptif untuk Standardisasi Data (Z-Score)
stats = {
    'Jenis Kelamin': {'mean': 0.56, 'std': 0.496},
    'Usia': {'mean': 38.85, 'std': 13.969},
    'Pendapatan Tahunan (juta Rp)': {'mean': 60.56, 'std': 26.264},
    'Skor Belanja (1-100)': {'mean': 50.20, 'std': 25.823},
    'Transaksi per Bulan': {'mean': 2.62, 'std': 1.423},
    'Rata-rata Nilai Transaksi (juta Rp)': {'mean': 2.478, 'std': 1.136},
    'Jumlah Kategori Produk Dibeli': {'mean': 1.235, 'std': 0.490},
    'Frekuensi Penggunaan Promo': {'mean': 2.075, 'std': 1.610},
    'Rata-rata Rating Diberikan': {'mean': 3.566, 'std': 0.534},
    'Metode Pembayaran Favorit': {'mean': 1.055, 'std': 1.002}
}

# 3. Deskripsi Detail untuk Masing-masing Segmen (Cluster)
cluster_info = {
    0: {
        'nama': 'Pelanggan Muda Promo Hunter',
        'deskripsi': 'Pelanggan berusia muda dengan pendapatan relatif rendah yang sangat aktif berbelanja terutama ketika ada promo/diskon. Nilai transaksi per belanja kecil namun frekuensinya cukup tinggi.',
        'icon': '🛍️',
        'warna_primary': '#ff6b6b',
        'warna_badge': 'rgba(255, 107, 107, 0.15)',
        'karakteristik': [
            'Usia rata-rata muda (~25 tahun)',
            'Pendapatan tahunan rendah (~26 Juta Rp)',
            'Skor belanja sangat tinggi (~75)',
            'Penggunaan promo sangat sering (rata-rata 4 kali per bulan)'
        ],
        'rekomendasi': 'Berikan penawaran diskon kilat (flash sale), promo cashback e-wallet, voucher gratis ongkir dengan batas minimum belanja rendah, dan produk-produk trendi dengan harga terjangkau.'
    },
    1: {
        'nama': 'Pelanggan Senior Hemat',
        'deskripsi': 'Pelanggan paruh baya hingga senior dengan pendapatan menengah ke bawah. Mereka cenderung berbelanja secara hati-hati, memprioritaskan kebutuhan pokok, dan menggunakan promo sewajarnya.',
        'icon': '👴',
        'warna_primary': '#fcc419',
        'warna_badge': 'rgba(252, 196, 25, 0.15)',
        'karakteristik': [
            'Usia rata-rata senior (~50 tahun)',
            'Pendapatan menengah ke bawah (~43 Juta Rp)',
            'Skor belanja cenderung rendah (~39)',
            'Frekuensi transaksi bulanan sedang (~2 kali per bulan)'
        ],
        'rekomendasi': 'Tawarkan promo paket bundling (misal kebutuhan pokok/kesehatan), diskon khusus usia emas, serta sediakan antarmuka aplikasi yang sederhana dan mudah dinavigasi.'
    },
    2: {
        'nama': 'Pelanggan Premium Loyal',
        'deskripsi': 'Pelanggan berusia produktif dengan pendapatan tinggi dan daya beli sangat kuat. Mereka sangat aktif berbelanja barang bernilai besar, sangat loyal, dan tidak terlalu peduli dengan kupon promo.',
        'icon': '💎',
        'warna_primary': '#339af0',
        'warna_badge': 'rgba(51, 154, 240, 0.15)',
        'karakteristik': [
            'Usia produktif dewasa muda (~31 tahun)',
            'Pendapatan tahunan tinggi (~78 Juta Rp)',
            'Skor belanja sangat tinggi (~73)',
            'Jarang menggunakan promo dan transaksi bernilai besar (~3.2 Juta Rp per transaksi)'
        ],
        'rekomendasi': 'Masukkan ke dalam program loyalitas eksklusif (VIP Club), tawarkan layanan prioritas, gratis ongkir instan tanpa syarat, akses awal (early access) untuk koleksi terbaru, dan penawaran personal berkualitas tinggi.'
    },
    3: {
        'nama': 'Pelanggan Kaya Pasif',
        'deskripsi': 'Pelanggan dengan pendapatan tinggi tetapi jarang melakukan transaksi. Sekalinya berbelanja, mereka membeli produk bernilai tinggi namun keterikatan (engagement) mereka terhadap marketplace masih rendah.',
        'icon': '💼',
        'warna_primary': '#94d82d',
        'warna_badge': 'rgba(148, 216, 45, 0.15)',
        'karakteristik': [
            'Usia dewasa matang (~40 tahun)',
            'Pendapatan tahunan tinggi (~82 Juta Rp)',
            'Skor belanja cenderung rendah (~25)',
            'Frekuensi belanja jarang (~1.3 kali) tetapi nilai transaksi besar (~3.3 Juta Rp)'
        ],
        'rekomendasi': 'Gunakan personal marketing email/notifikasi tentang barang-barang mewah (high-end), rilis produk premium terbaru, serta promo eksklusif akhir pekan untuk menarik kembali minat belanja mereka.'
    },
    4: {
        'nama': 'Pelanggan Aktif Eksploratif',
        'deskripsi': 'Pelanggan sangat aktif dengan pendapatan menengah. Mereka sangat gemar mengeksplorasi dan membeli berbagai jenis kategori produk berbeda serta memiliki tingkat kepuasan yang tinggi.',
        'icon': '🚀',
        'warna_primary': '#ae3ec9',
        'warna_badge': 'rgba(174, 62, 201, 0.15)',
        'karakteristik': [
            'Usia dewasa muda (~34 tahun)',
            'Pendapatan tahunan menengah (~61 Juta Rp)',
            'Membeli banyak kategori produk berbeda (rata-rata > 2 kategori)',
            'Sangat aktif berbelanja (~4 kali) dengan tingkat kepuasan yang tinggi (~3.84/5)'
        ],
        'rekomendasi': 'Gunakan sistem rekomendasi produk (recommender system) lintas kategori (cross-selling), tawarkan reward poin berdasarkan keaktifan harian/mingguan, serta berikan penawaran paket produk serbaguna.'
    }
}

@app.route('/', methods=['GET', 'POST'])
def home():
    hasil_cluster = None
    info_cluster = None
    if request.method == 'POST':
        # Ambil data langsung dari form HTML
        gender = int(request.form['jenis_kelamin'])
        usia = float(request.form['usia'])
        pendapatan = float(request.form['pendapatan'])
        skor_belanja = float(request.form['skor_belanja'])
        transaksi = float(request.form['transaksi_per_bulan'])
        rata_transaksi = float(request.form['rata_nilai_transaksi'])
        kategori = float(request.form['jumlah_kategori'])
        promo = float(request.form['frekuensi_promo'])
        rating = float(request.form['rata_rating'])
        pembayaran = int(request.form['metode_pembayaran'])

        # Susun data mentah sesuai urutan fitur
        raw_values = [gender, usia, pendapatan, skor_belanja, transaksi, rata_transaksi, kategori, promo, rating, pembayaran]
        
        # Scaling manual Z-Score agar performa web instan
        keys = list(stats.keys())
        scaled_features = []
        for i, val in enumerate(raw_values):
            feature_name = keys[i]
            scaled_val = (val - stats[feature_name]['mean']) / stats[feature_name]['std']
            scaled_features.append(scaled_val)

        # Prediksi cluster
        input_scaled = np.array([scaled_features])
        hasil_cluster = int(model_kmeans.predict(input_scaled)[0])
        info_cluster = cluster_info.get(hasil_cluster)

    return render_template('index.html', hasil=hasil_cluster, info=info_cluster)

if __name__ == '__main__':
    app.run(debug=True, port=5001)