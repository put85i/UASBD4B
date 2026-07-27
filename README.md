# Big Data Watch-Time Forecasting Engine
**Topik 1 (Regresi) — UAS Praktikum Big Data, Universitas Bale Bandung (UNIBBA)**
Prodi Teknik Informatika / Sistem Informasi · Dosen: Sutiyono, M.Kom · UAS Genap TA 2025/2026

## 1. Ringkasan

Proyek ini memprediksi durasi menonton (`play_duration_clean`) sebagai target kontinu dari data log platform video "videodotcom", menggunakan **Apache Spark MLlib** di atas cluster **Docker + HDFS + Spark Standalone**.

| Item | Detail |
|---|---|
| Target | `play_duration_clean` (kontinu, hasil casting dari `play_duration`) |
| Fitur prediktor | `category_vec`, `platform_vec`, `os_vec`, `is_premium` |
| Algoritma | `LinearRegression` (Spark ML) |
| Evaluasi | RMSE, MAE, R² Score |
| Dataset | `videodotcom_big.csv` — **15.611.672.030 byte (≈14,5 GiB)**, HDFS `/shared/data/` |
| Baris mentah | 23.743.309 |
| Data latih / uji | 18.993.082 / 4.750.227 (split 80:20, seed 42) |

## 2. Arsitektur Cluster

```
Docker Compose
├── namenode_selsa          (HDFS NameNode)
├── datanode_selsa          (HDFS DataNode)
├── spark-master_selsa      (Spark Standalone Master)
├── spark-worker-1_selsa    (Spark Executor)
└── spark-worker-2_selsa    (Spark Executor, 4 core / worker)
```

Job dijalankan dalam mode `client` ke `spark://spark-master:7077`, dengan skrip disalin ke `spark-master_selsa:/opt/spark/work-dir/ml_regression.py` sebelum `spark-submit`.

## 3. Pipeline Data (Bab 2 Laporan)

1. **Ingestion** — baca `videodotcom_big.csv` dari HDFS via `SparkSession.read.csv(inferSchema=True)`.
2. **Casting & Handling Null** — `play_duration` di-cast ke `IntegerType` → `play_duration_clean`, null diisi `0`. Kolom `is_premium` (boolean, ada null) di-cast ke `double` lalu `na.fill(0.0)` sebelum masuk `VectorAssembler`.
3. **Audit Anti-Data Leakage** — kolom `play_duration` (sumber langsung target) dan `is_active_session` (turunan target) **sengaja dikeluarkan** dari fitur prediktor. Fitur yang dipakai hanya `category_name`, `platform`, `os_name`, `is_premium`.
4. **Feature Engineering** — `StringIndexer` → `OneHotEncoder` untuk `category_name`, `platform`, `os_name`, lalu digabung dengan `is_premium` via `VectorAssembler(handleInvalid="skip")` menjadi kolom `features`.

## 4. Modeling (Bab 3 Laporan)

Seluruh tahapan dirangkai dalam satu Spark `Pipeline`:

```
[StringIndexer x3] -> [OneHotEncoder x3] -> [VectorAssembler] -> [LinearRegression]
```

Data displit 80:20 (`seed=42`) menjadi data latih dan data uji, lalu pipeline di-`fit()` pada data latih dan dievaluasi pada data uji.

## 5. Hasil Evaluasi (Bab 4 Laporan)

Hasil aktual dari run `spark-submit` (log lengkap: `terminal_uas_big_data.txt`):

| Metrik | Nilai |
|---|---|
| RMSE | 1551.998957 |
| MAE | 543.489720 |
| R² | 0.082640 |

Grafik residual plot (`prediction` vs `residual`, sampel 5%) diekspor ke `grafik_residual_plot.png`.

**Interpretasi:** R² ≈ 0,08 tergolong rendah — artinya kombinasi fitur kategorikal (`category`, `platform`, `os`) dan status `is_premium` saja **belum cukup** menjelaskan variasi durasi tonton pengguna. Model cenderung menghasilkan prediksi yang mengelompok di sekitar rata-rata per kombinasi fitur kategorikal, sehingga error tetap tinggi untuk baris dengan durasi tonton ekstrem (0 atau sangat lama). Ini jadi dasar rekomendasi di Bab 5.

## 6. Rekomendasi Strategi & Aksi Bisnis (Bab 5 Laporan)

- **Tambah fitur perilaku**: riwayat tontonan (watch history), waktu akses (jam/hari), durasi sesi sebelumnya, dan jenis konten yang sering diulang — fitur ini biasanya jauh lebih prediktif untuk watch-time dibanding atribut statis pengguna/perangkat.
- **Coba algoritma non-linear**: `DecisionTreeRegressor` atau `GBTRegressor` berpotensi menangkap interaksi antar fitur kategorikal yang tidak tertangkap regresi linear.
- **Segmentasi dulu, prediksi kemudian**: hasil Topik 4 (K-Means segmentation) bisa dipakai sebagai fitur tambahan atau untuk melatih model terpisah per segmen pengguna.
- **Evaluasi ulang skala target**: distribusi `play_duration_clean` kemungkinan sangat skewed (banyak nilai 0); pertimbangkan transformasi log atau model khusus zero-inflated.

## 7. Cara Menjalankan Ulang

```bash
# 1. Nyalakan container yang dibutuhkan
docker compose up -d namenode datanode spark-master spark-worker-1 spark-worker-2

# 2. Pastikan HDFS & dataset siap
docker exec -it namenode_selsa hdfs dfs -ls /shared/data

# 3. Copy skrip ke container Spark master
docker cp ml_regression.py spark-master_selsa:/opt/spark/work-dir/

# 4. Jalankan training (pakai path lengkap, spark-submit tidak ada di $PATH image ini)
docker exec -it spark-master_selsa /spark/bin/spark-submit \
  --master spark://spark-master:7077 --deploy-mode client \
  /opt/spark/work-dir/ml_regression.py

# 5. Ambil hasil residual plot (tersimpan di root container, path relatif)
docker exec -it spark-master_selsa find / -name "grafik_residual_plot.png"
docker cp spark-master_selsa:/grafik_residual_plot.png ./
```

> ⏱️ Proses training pada dataset ~14,5 GiB memakan waktu cukup lama dengan 2 worker (4 core/worker). Disarankan dijalankan jauh-jauh hari sebelum demo dan hasilnya didokumentasikan lewat screenshot/screen-recording.

## 8. Riwayat Error & Solusi (Debugging Log)

| No | Error | Penyebab | Solusi |
|---|---|---|---|
| 1 | `Error response from daemon: No such container` | Nama container sebenarnya memakai suffix `_selsa` (mis. `namenode_selsa`), bukan nama polos `namenode` | Gunakan nama container sesuai `docker compose ps`, mis. `namenode_selsa`, `spark-master_selsa` |
| 2 | `spark-submit: executable file not found in $PATH` | Binary `spark-submit` tidak ada di `$PATH` image ini | Panggil pakai path lengkap `/spark/bin/spark-submit` |
| 3 | `Encountered null while assembling a row with handleInvalid="error"` di `VectorAssembler` | Kolom `is_premium` (boolean) mengandung null di data mentah | Cast ke `double` + `na.fill({"is_premium": 0.0})`, tambah `handleInvalid="skip"` di `VectorAssembler` |
| 4 | `docker cp` gagal, file `grafik_residual_plot.png` tidak ditemukan di `/opt/spark/work-dir/` | `plt.savefig()` memakai path relatif → tersimpan di working directory proses Spark (root container), bukan folder skrip | Cari dulu dengan `find / -name "grafik_residual_plot.png"`, baru `docker cp` sesuai path yang ditemukan |
| 5 | `Cholesky solver failed due to singular covariance matrix` (WARNING, bukan error fatal) | Matriks fitur hasil One-Hot Encoding bersifat sparse/singular | Spark otomatis fallback ke Quasi-Newton solver, training tetap lanjut dan selesai normal |

## 9. Struktur File

```
.
├── ml_regression.py            # Skrip PySpark end-to-end (ingestion → cleaning → pipeline → evaluasi → plot)
├── terminal_uas_big_data.txt   # Log lengkap eksekusi spark-submit (bukti hasil aktual)
├── grafik_residual_plot.png    # Output visualisasi residual plot
└── README.md                   # Dokumen ini
```

