# Naskah Presentasi (slide demi slide)

**Slide 1 – Judul**  
“Halo, saya Muhammad Razan Alamudi. Ini adalah ‘Wi-Fi AP Clustering with Deterministic Annealing’—dataset ‘data 3 access point,’ 287 AP, hanya pandas + numpy.”

**Slide 2 – Masalah & Tujuan (kenapa tim ops peduli)**  
“Tim ops perlu tahu AP mana yang panas, stabil, atau kurang terpakai. Tujuannya bikin keranjang sederhana supaya kita bisa memberi relief, memantau, atau memanfaatkan ulang perangkat. Batasan: tanpa pustaka ML, jadi kami mengoptimalkan objektif k-means dasar dengan metode niche.”

**Slide 3 – Data & Fitur**  
“Kami ambil lima CSV ter-gzip: clients, CPU, memory, dan sinyal 2.4G serta 5G. Itu menghasilkan 13 fitur per AP—statistik beban, CPU/memory, dan sinyal min/mean/max di kedua band. NaN pada client-std (seri konstan) diset ke 0. Setelah imputasi ada 287 AP dengan fitur terstandardisasi.”

**Slide 4 – Praproses**  
“Kami isi metrik yang hilang dengan rata-rata fitur sehingga tidak ada AP yang dibuang. Lalu semua fitur di-z-score supaya tidak ada metrik yang mendominasi. Cek integritas: bentuk 287×13, tidak ada NaN, dan `ap_name` disimpan untuk interpretasi.”

**Slide 5 – Metode yang Dibandingkan (bahasa sederhana)**  
“Dua metode pengelompokan untuk objektif yang sama:  

1) Baseline: jalankan k-means++ berkali-kali di k=3, wajib minimal 5 AP per klaster, simpan pemisahan dengan separasi terbaik.  
2) Optimizer: deterministic annealing—mulai dengan penugasan fuzzy yang mengeras saat pendinginan; k=3 tetap; simpan klaster paling rapat. Seed: 99 dan 123.”

**Slide 6 – Cara kami menilai hasil**  
“Dua skor: SSE (kekompakan di dalam klaster—lebih rendah lebih baik) dan silhouette (separasi antar klaster—lebih tinggi lebih baik). Hasil: Baseline k=3 → SSE 2803.64, silhouette 0.212, ukuran [86, 25, 176]. Annealing k=3 → SSE 2731.39, silhouette 0.166, ukuran [88, 82, 117]. Trade-off: baseline lebih jelas memisah; annealing lebih rapat tapi lebih tumpang tindih.”

**Slide 7 – Narasi Klaster Baseline (k=3)**  
“Tiga klaster:  
- Panas (86): ~3.2 klien, CPU/memory lebih tinggi, sinyal ~ -63/-58 dBm → beri relief/perkuat.  
- Sinyal-kuat beban-sedang (25): ~2.2 klien, CPU/memory seimbang, sinyal ~ -59/-58 dBm → pantau/ukuran tepat.  
- Kurang terpakai (176): ~0.7 klien, sinyal ~ -65/-58 dBm → manfaatkan ulang/retuning.”

**Slide 8 – Narasi Klaster Annealing (k=3)**  
“Tiga tingkat:  
- Panas (88): ~3.4 klien, CPU/memory lebih tinggi, sinyal ~ -63/-57 dBm → beri relief/perkuat.  
- Memori-tinggi (82): ~1 klien, memori ~0.57, sinyal ~ -63/-55 dBm → audit/optimasi.  
- Kurang terpakai/sinyal-lemah (117): ~0.6 klien, sinyal ~ -65/-61 dBm → manfaatkan ulang/retuning.”

**Slide 9 – Visual (apa yang ditunjukkan plot)**  
“Berikut plot PCA 2D. Tiap titik adalah AP yang diproyeksikan ke 2D; warna = klaster; X menandai centroid. Kiri: baseline k=3 (`baseline_clusters.png`, judul: ‘Baseline k=3 (PCA 2D)’). Kanan: annealing k=3 (`annealing_clusters.png`, judul: ‘Deterministic Annealing k=3 (PCA 2D)’). Baseline sedikit lebih bersih memisah; annealing lebih merapatkan klaster.”

**Slide 10 – Rekomendasi aksi**  
“Gunakan label k=3 untuk playbook: beri relief pada panas, audit memori-tinggi/sinyal-kuat, manfaatkan ulang/retuning yang kurang terpakai. Pilih label baseline untuk separasi lebih jelas; pilih label annealing untuk SSE lebih rapat dan tier memori-tinggi yang menonjol.”

**Slide 11 – Keterbatasan & Langkah Lanjut**  
“Batas: tidak ada metrik interferensi/channel; k-means mengasumsikan klaster sferis; silhouette melembut karena semua AP dipertahankan lewat imputasi. Berikutnya: tambah fitur RF/interferensi, uji lebih banyak seed/perturbasi, tinjau ulang k, dan coba metode non-sferis.”

**Slide 12 – Lampiran**  
“Hyperparameter bagi yang penasaran: annealing T0=6.0, alpha=0.9, inner steps=12; baseline restarts=120, ukuran klaster min=5. Ada juga histori SSE untuk annealing dan catatan bahwa menghapus baris vs. imputasi mengubah jumlah dan skor.”
