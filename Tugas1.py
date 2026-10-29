"""Generator dataset simulasi Ethercraft (200 baris transaksi).
Seed tetap agar hasil dapat direproduksi. Data sepenuhnya sintetis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(42)
N, N_CUST = 200, 120

# ---------- Tabel pelanggan (atribut tetap per pelanggan) ----------
segments = ["Pemula", "Kompetitif PvP", "Kreator Konten"]
cities = ["Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Medan", "Makassar"]
cust = pd.DataFrame({
    "customer_id": [f"CUST{i:03d}" for i in range(1, N_CUST + 1)],
    "age": rng.integers(15, 31, size=N_CUST),
    "gender": rng.choice(["Laki-laki", "Perempuan"], size=N_CUST, p=[0.75, 0.25]),
    "city": rng.choice(cities, size=N_CUST, p=[0.3, 0.15, 0.15, 0.15, 0.1, 0.15]),
    "player_segment": rng.choice(segments, size=N_CUST, p=[0.45, 0.35, 0.2]),
})
hours_by_seg = {"Pemula": 60, "Kompetitif PvP": 450, "Kreator Konten": 300}
cust["hours_played"] = [
    max(5, int(rng.normal(hours_by_seg[s], hours_by_seg[s] * 0.35))) for s in cust["player_segment"]
]

# ---------- Produk ----------
base_price = {"Build Guide": 49_000, "Coaching": 250_000, "Apparel": 175_000,
              "Poster": 85_000, "Aset Kreator": 99_000}
cat_probs = {  # kebutuhan produk bergantung pada segmen pemain
    "Pemula":         [0.55, 0.20, 0.10, 0.10, 0.05],
    "Kompetitif PvP": [0.30, 0.30, 0.25, 0.10, 0.05],
    "Kreator Konten": [0.10, 0.10, 0.25, 0.10, 0.45],
}
categories = list(base_price)

# ---------- Transaksi ----------
cids = rng.choice(cust["customer_id"], size=N)
df = pd.DataFrame({"order_id": [f"ETH{1000 + i}" for i in range(N)], "customer_id": cids})
df = df.merge(cust, on="customer_id", how="left")

df["product_category"] = [rng.choice(categories, p=cat_probs[s]) for s in df["player_segment"]]
price = np.array([base_price[c] for c in df["product_category"]]) * rng.uniform(0.9, 1.2, size=N)
df["unit_price"] = ((price / 500).round() * 500).astype(float)
digital = df["product_category"].isin(["Build Guide", "Coaching", "Aset Kreator"])
df["quantity"] = np.where(digital, 1, rng.choice([1, 2, 3], size=N, p=[0.7, 0.2, 0.1]))
df["discount_pct"] = rng.choice([0, 5, 10, 15, 20], size=N, p=[0.45, 0.15, 0.2, 0.1, 0.1])
df["total_amount"] = (df["unit_price"] * df["quantity"] * (1 - df["discount_pct"] / 100)).round()

df["order_date"] = (pd.to_datetime("2026-01-01")
                    + pd.to_timedelta(rng.integers(0, 270, size=N), unit="D")).strftime("%Y-%m-%d")
df["game_version"] = np.where(pd.to_datetime(df["order_date"]) < "2026-05-01", "Chapter 1", "Chapter 2")
df["payment_method"] = rng.choice(["QRIS", "E-Wallet", "Transfer Bank", "Kartu Kredit"],
                                  size=N, p=[0.35, 0.35, 0.2, 0.1])
df["traffic_source"] = rng.choice(["TikTok", "YouTube", "Discord", "Instagram", "Direct"],
                                  size=N, p=[0.3, 0.25, 0.25, 0.1, 0.1])
df["rating"] = np.clip(np.round(rng.normal(4.3, 0.8, size=N)), 1, 5)

pos = {
    "Build Guide": ["Panduan jelas, build langsung bisa dipakai", "Sangat membantu untuk pemula"],
    "Coaching": ["Sesi coaching padat dan sabar", "Skill parry saya naik drastis"],
    "Apparel": ["Bahan nyaman, desain keren", "Sablon rapi dan awet"],
    "Poster": ["Cetakan tajam, warna bagus", "Cocok untuk dekor setup"],
    "Aset Kreator": ["Overlay rapi, mudah dipasang", "Ikon Discord bagus"],
}
neg = ["Pengiriman agak lama", "Build sudah kurang relevan setelah update",
       "Ukuran tidak sesuai harapan", "Jadwal coaching sulit disesuaikan"]
df["review_text"] = [rng.choice(pos[c]) if r >= 4 else rng.choice(neg)
                     for c, r in zip(df["product_category"], df["rating"])]

df["is_repeat_customer"] = df.duplicated("customer_id", keep=False).astype(int)

# ---------- Missing values (realistis untuk data mentah) ----------
def inject_nan(col, frac):
    idx = rng.choice(df.index, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

inject_nan("age", 0.08)
inject_nan("hours_played", 0.10)
inject_nan("city", 0.05)
inject_nan("payment_method", 0.04)
inject_nan("rating", 0.15)        # tidak semua pembeli memberi rating
inject_nan("review_text", 0.30)   # banyak yang tidak menulis ulasan

# Satu outlier sengaja: pesanan apparel massal untuk guild (untuk latihan EDA)
i = df.index[df["product_category"] == "Apparel"][0]
df.loc[i, "quantity"] = 30
df.loc[i, "total_amount"] = df.loc[i, "unit_price"] * 30 * (1 - df.loc[i, "discount_pct"] / 100)

cols = ["order_id", "customer_id", "order_date", "age", "gender", "city", "player_segment",
        "hours_played", "product_category", "unit_price", "quantity", "discount_pct",
        "total_amount", "payment_method", "traffic_source", "game_version", "rating",
        "review_text", "is_repeat_customer"]
df[cols].to_csv("ethercraft_transactions_raw.csv", index=False)
print(df[cols].shape)
print(df[cols].isna().sum()[df[cols].isna().sum() > 0])

# ---------- Kamus data ----------
data_dictionary = pd.DataFrame([
    ("order_id", "Kategorikal", "ID unik transaksi."),
    ("customer_id", "Kategorikal", "ID pelanggan sintetis."),
    ("order_date", "Tanggal", "Tanggal transaksi dalam format YYYY-MM-DD."),
    ("age", "Numerik", "Usia pelanggan dalam tahun; dapat kosong."),
    ("gender", "Kategorikal", "Kategori gender pelanggan."),
    ("city", "Kategorikal", "Kota tempat tinggal pelanggan; dapat kosong."),
    ("player_segment", "Kategorikal", "Segmen pemain pelanggan."),
    ("hours_played", "Numerik", "Perkiraan total jam bermain; dapat kosong."),
    ("product_category", "Kategorikal", "Kategori produk yang dibeli."),
    ("unit_price", "Numerik", "Harga satuan produk dalam rupiah."),
    ("quantity", "Numerik", "Jumlah produk dalam transaksi."),
    ("discount_pct", "Numerik", "Persentase diskon transaksi."),
    ("total_amount", "Numerik", "Total pembayaran setelah diskon, dalam rupiah."),
    ("payment_method", "Kategorikal", "Metode pembayaran; dapat kosong."),
    ("traffic_source", "Kategorikal", "Kanal yang menjadi sumber kunjungan."),
    ("game_version", "Kategorikal", "Versi game saat transaksi."),
    ("rating", "Numerik", "Rating pelanggan dari 1 sampai 5; dapat kosong."),
    ("review_text", "Teks", "Ulasan pelanggan; dapat kosong."),
    ("is_repeat_customer", "Biner", "1 jika pelanggan muncul lebih dari sekali, selain itu 0."),
], columns=["Variabel", "Tipe", "Deskripsi"])
print("\nKamus Data:")
print(data_dictionary.to_string(index=False))
data_dictionary.to_csv("ethercraft_data_dictionary.csv", index=False)

# ---------- Pembersihan dan preprocessing ----------
clean_df = df[cols].copy()
clean_df["review_text"] = clean_df["review_text"].fillna("Tidak ada ulasan")

numeric_features = ["age", "hours_played", "unit_price", "quantity", "discount_pct", "rating"]
categorical_features = [
    "gender", "city", "player_segment", "product_category", "payment_method",
    "traffic_source", "game_version", "is_repeat_customer",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numerik", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("kategori", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ],
    verbose_feature_names_out=False,
)

transformed = preprocessor.fit_transform(clean_df)
if hasattr(transformed, "toarray"):
    transformed = transformed.toarray()

feature_names = preprocessor.get_feature_names_out()
processed_features = pd.DataFrame(transformed, columns=feature_names, index=clean_df.index)
processed_df = pd.concat([
    clean_df[["order_id", "customer_id", "order_date", "review_text"]],
    processed_features,
], axis=1)
processed_df.to_csv("ethercraft_transactions_processed.csv", index=False)
print("\nData hasil preprocessing:", processed_df.shape)
print("Missing values pada fitur hasil preprocessing:", int(processed_features.isna().sum().sum()))

# ---------- Machine learning: prediksi omzet transaksi ----------
# Harga, jumlah, dan diskon tidak digunakan agar target tidak bocor ke fitur.
model_numeric_features = ["age", "hours_played"]
model_categorical_features = [
    "gender", "city", "player_segment", "product_category", "traffic_source", "game_version",
]
model_features = model_numeric_features + model_categorical_features
X = df[model_features].copy()
y = df["total_amount"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_preprocessor = ColumnTransformer(
    transformers=[
        ("numerik", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), model_numeric_features),
        ("kategori", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), model_categorical_features),
    ]
)

model = Pipeline([
    ("preprocessing", model_preprocessor),
    ("regressor", LinearRegression()),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nEvaluasi model regresi (data uji):")
print(f"Jumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")
print(f"MAE : Rp{mae:,.0f}")
print(f"RMSE: Rp{rmse:,.0f}")
print(f"R²  : {r2:.3f}")

# Grafik prediksi dibandingkan nilai transaksi sebenarnya.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.75, color="#176b87", edgecolor="white")
axis_min = min(y_test.min(), y_pred.min())
axis_max = max(y_test.max(), y_pred.max())
plt.plot([axis_min, axis_max], [axis_min, axis_max], "--", color="#d1495b", label="Prediksi ideal")
plt.xlabel("Omzet aktual (Rupiah)")
plt.ylabel("Omzet prediksi (Rupiah)")
plt.title("Omzet Aktual vs Prediksi - Data Uji")
plt.legend()
plt.tight_layout()
plt.savefig("ethercraft_actual_vs_predicted.png", dpi=150)
plt.close()

# Prediksi contoh transaksi baru, sebelum nilai omzet diketahui.
new_order = pd.DataFrame([{
    "age": 22,
    "hours_played": 140,
    "gender": "Perempuan",
    "city": "Jakarta",
    "player_segment": "Kompetitif PvP",
    "product_category": "Coaching",
    "traffic_source": "YouTube",
    "game_version": "Chapter 2",
}])
new_order_prediction = model.predict(new_order[model_features])[0]
print(f"\nPrediksi omzet transaksi baru (contoh): Rp{new_order_prediction:,.0f}")

# Insight memakai prediksi pada data uji, bukan hasil training.
test_results = X_test[["product_category"]].copy()
test_results["predicted_amount"] = y_pred
category_insights = test_results.groupby("product_category").agg(
    jumlah_transaksi_uji=("predicted_amount", "size"),
    rata_rata_omzet_prediksi=("predicted_amount", "mean"),
).sort_values("rata_rata_omzet_prediksi", ascending=False)
print("\nRata-rata omzet prediksi per kategori (data uji):")
print(category_insights.round(0).to_string())

highest_category = category_insights.index[0]
lowest_category = category_insights.index[-1]
print("\nRekomendasi bisnis:")
print(
    f"1. Uji kampanye yang menonjolkan {highest_category}, kategori dengan rata-rata "
    f"omzet prediksi tertinggi pada data uji (Rp{category_insights.iloc[0]['rata_rata_omzet_prediksi']:,.0f} "
    f"per transaksi). Pantau margin dan konversi sebelum menaikkan anggaran."
)
print(
    f"2. Uji bundling atau upsell untuk {lowest_category}, kategori dengan rata-rata "
    f"omzet prediksi terendah pada data uji (Rp{category_insights.iloc[-1]['rata_rata_omzet_prediksi']:,.0f} "
    f"per transaksi), lalu bandingkan hasilnya dengan transaksi tanpa promosi."
)