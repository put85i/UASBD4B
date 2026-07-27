"""
Topik 1 (Regresi) - Big Data Watch-Time Forecasting Engine
UAS Praktikum Big Data - UNIBBA
Target: play_duration_clean (kontinu)
Fitur: category_vec, platform_vec, os_vec, is_premium
Algoritma: LinearRegression (bisa diganti DecisionTreeRegressor)
Evaluasi: RMSE, MAE, R2 Score
Dataset: videodotcom_big.csv (4.5 GB, HDFS /shared/data/)
"""

import matplotlib
matplotlib.use("Agg")  # headless mode, wajib sebelum import pyplot
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# =========================================================
# TAHAPAN 0: INISIALISASI SPARK SESSION
# =========================================================
spark = SparkSession.builder \
    .appName("BigDataWatchTimeForecastingEngine") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# =========================================================
# TAHAPAN 1: INGESTION DATA DARI HDFS
# =========================================================
DATA_PATH = "hdfs://namenode:9000/shared/data/videodotcom_big.csv"

df_raw = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

print("Skema awal dataset:")
df_raw.printSchema()
print(f"Jumlah baris mentah: {df_raw.count()}")

# =========================================================
# TAHAPAN 2: DATA CLEANING (CASTING, HANDLING NULL)
# =========================================================
df_clean = df_raw.withColumn(
    "play_duration_clean",
    col("play_duration").cast(IntegerType())
).na.fill({"play_duration_clean": 0})

# =========================================================
# AUDIT FEATURE LEAKAGE (WAJIB didokumentasikan di Bab 3 laporan)
# =========================================================
# Kolom play_duration (mentah) dan is_active_session (label turunan dari
# play_duration di Modul 14) DILARANG masuk sebagai fitur prediktor, karena
# keduanya adalah sumber/turunan langsung dari target play_duration_clean.
# Fitur yang benar-benar dipakai sebagai prediktor hanya:
#   category_vec, platform_vec, os_vec, is_premium
LEAKY_COLUMNS = ["play_duration", "is_active_session"]
ALLOWED_FEATURE_SOURCE_COLS = ["category_name", "platform", "os_name", "is_premium"]

print("=== AUDIT ANTI-DATA LEAKAGE ===")
print(f"Kolom yang DIKELUARKAN dari VectorAssembler (sumber/turunan target): {LEAKY_COLUMNS}")
print(f"Kolom yang DIPAKAI sebagai fitur prediktor: {ALLOWED_FEATURE_SOURCE_COLS}")

# =========================================================
# TAHAPAN 3: FEATURE ENGINEERING
# StringIndexer -> OneHotEncoder -> VectorAssembler
# =========================================================
categorical_cols = ["category_name", "platform", "os_name"]

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
    for c in categorical_cols
]

# is_premium wajib ikut sebagai fitur prediktor sesuai ketentuan Modul Teknis
# Topik 1. Kolom ini boolean dan bisa berisi null di data mentah, jadi wajib
# di-cast ke double DULU baru di-fillna, supaya tidak ada null yang lolos
# ke VectorAssembler (kalau tidak, job akan gagal saat fit() setelah
# memproses jutaan baris, buang banyak waktu komputasi).
df_clean = df_clean.withColumn("is_premium", col("is_premium").cast("double"))
df_clean = df_clean.na.fill({"is_premium": 0.0})

assembler = VectorAssembler(
    inputCols=[f"{c}_vec" for c in categorical_cols] + ["is_premium"],
    outputCol="features",
    handleInvalid="skip"  # jaring pengaman kalau masih ada null tak terduga
)

# =========================================================
# TAHAPAN 4: MODELING - LINEAR REGRESSION DALAM PIPELINE
# =========================================================
lr = LinearRegression(
    featuresCol="features",
    labelCol="play_duration_clean",
    predictionCol="prediction"
)

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

# Split data 80:20, seed konsisten dengan modul sebelumnya
train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)

print(f"Jumlah baris data latih: {train_df.count()}")
print(f"Jumlah baris data uji  : {test_df.count()}")

model = pipeline.fit(train_df)
predictions = model.transform(test_df)

# =========================================================
# TAHAPAN 5: OUTPUT BERTAHAP (STAGE LOGGING)
# =========================================================
print("\n=== Tahapan 1: Hasil StringIndexer ===")
model.stages[0].transform(train_df).select(categorical_cols[0], f"{categorical_cols[0]}_idx").show(5)

print("\n=== Tahapan 2: Hasil OneHotEncoder ===")
predictions.select([f"{c}_vec" for c in categorical_cols]).show(5, truncate=False)

print("\n=== Tahapan 3: Hasil VectorAssembler ===")
predictions.select("features").show(5, truncate=False)

print("\n=== Tahapan 4: Hasil Akhir Prediksi ===")
predictions.select("play_duration_clean", "prediction").show(10)

# =========================================================
# TAHAPAN 6: EVALUASI - RMSE DAN R2
# =========================================================
evaluator_rmse = RegressionEvaluator(
    labelCol="play_duration_clean", predictionCol="prediction", metricName="rmse"
)
evaluator_mae = RegressionEvaluator(
    labelCol="play_duration_clean", predictionCol="prediction", metricName="mae"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="play_duration_clean", predictionCol="prediction", metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"\n=== REKAPITULASI METRIK EVALUASI ===")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R2   : {r2:.6f}")

# =========================================================
# TAHAPAN 7: RESIDUAL PLOT (untuk Bab IV laporan)
# =========================================================
sample_pd = predictions.select("play_duration_clean", "prediction") \
    .sample(fraction=0.05, seed=42).toPandas()
sample_pd["residual"] = sample_pd["play_duration_clean"] - sample_pd["prediction"]

plt.figure(figsize=(8, 6))
plt.scatter(sample_pd["prediction"], sample_pd["residual"], alpha=0.4, s=10)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted play_duration_clean")
plt.ylabel("Residual")
plt.title("Residual Plot - Big Data Watch-Time Forecasting Engine")
plt.savefig("grafik_residual_plot.png", dpi=150, bbox_inches="tight")
print("\nResidual plot tersimpan: grafik_residual_plot.png")

spark.stop()