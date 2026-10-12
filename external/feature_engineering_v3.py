import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

print("=" * 100)
print("PRC_4 — ADVANCED FEATURE GENERATOR V3 (CDM TIMING + RUNWAY PHYSICALS + HIERARCHICAL TE)")
print("Hedef: Model A RMSE'sini 246s'den 213s seviyesine indirecek fiziksel ve zamanlama özellikleri")
print("=" * 100)

t0 = time.time()
BASE_PRC4 = os.environ["PRC4_DIR"]
BASE_PRC2 = os.environ["PRC2_DIR"]

TRAIN_PATH = os.path.join(BASE_PRC2, "data_2", "processed", "dep_train.parquet")
RANK_PATH  = os.path.join(BASE_PRC2, "data_2", "processed", "dep_rank.parquet")
RWY_PATH   = os.path.join(BASE_PRC4, "data", "features", "external_runways.csv")

OUT_FEAT_TRAIN = os.path.join(BASE_PRC4, "data", "features", "advanced_features_train_v3.parquet")
OUT_FEAT_RANK  = os.path.join(BASE_PRC4, "data", "features", "advanced_features_rank_v3.parquet")

print("\n[1/4] Veriler Yükleniyor...")
df_train = pd.read_parquet(TRAIN_PATH, columns=["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", 
                                                "AOBT_DIFF_SEC", "EOBT_DIFF_SEC", "LOBT_DIFF_SEC", 
                                                "IOBT_DIFF_SEC", "ARVT3_DIFF_SEC", "SCHED_DIFF_SEC", 
                                                "NM_TAXI_DIFF_SEC", "TAXITIME_SEC_mvt", "NM_OFFSET_SEC"])
df_rank = pd.read_parquet(RANK_PATH, columns=["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", 
                                              "AOBT_DIFF_SEC", "EOBT_DIFF_SEC", "LOBT_DIFF_SEC", 
                                              "IOBT_DIFF_SEC", "ARVT3_DIFF_SEC", "SCHED_DIFF_SEC", 
                                              "NM_TAXI_DIFF_SEC"])

print(f"  Eğitim: {len(df_train):,} satır | Test: {len(df_rank):,} satır")

# -------------------------------------------------------------
# 1. RUNWAY FİZİKSEL VERİLERİNİ EŞLEŞTİRME
# -------------------------------------------------------------
print("\n[2/4] Pist Fiziksel Boyutları (Uzunluk, Yön, Eşik) Entegre Ediliyor...")
df_rwy = pd.read_csv(RWY_PATH)
airports = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
sub_rwy = df_rwy[df_rwy['airport_ident'].isin(airports)].copy()

rwy_len_map = {}
rwy_head_map = {}
rwy_disp_map = {}

for idx, r in sub_rwy.iterrows():
    apt = r['airport_ident']
    length = r['length_ft']
    if pd.notna(r['le_ident']):
        ident = str(r['le_ident']).strip()
        rwy_len_map[(apt, ident)] = length
        rwy_head_map[(apt, ident)] = r['le_heading_degT']
        rwy_disp_map[(apt, ident)] = r['le_displaced_threshold_ft']
        if ident.startswith('0'):
            rwy_len_map[(apt, ident[1:])] = length
            rwy_head_map[(apt, ident[1:])] = r['le_heading_degT']
            rwy_disp_map[(apt, ident[1:])] = r['le_displaced_threshold_ft']
    if pd.notna(r['he_ident']):
        ident = str(r['he_ident']).strip()
        rwy_len_map[(apt, ident)] = length
        rwy_head_map[(apt, ident)] = r['he_heading_degT']
        rwy_disp_map[(apt, ident)] = r['he_displaced_threshold_ft']
        if ident.startswith('0'):
            rwy_len_map[(apt, ident[1:])] = length
            rwy_head_map[(apt, ident[1:])] = r['he_heading_degT']
            rwy_disp_map[(apt, ident[1:])] = r['he_displaced_threshold_ft']

def get_rwy_features(df):
    r_len = np.array([rwy_len_map.get((apt, str(rwy).strip()), np.nan) for apt, rwy in zip(df['ADEP_mvt'], df['RUNWAY_mvt'])], dtype=np.float32)
    r_head = np.array([rwy_head_map.get((apt, str(rwy).strip()), np.nan) for apt, rwy in zip(df['ADEP_mvt'], df['RUNWAY_mvt'])], dtype=np.float32)
    r_disp = np.array([rwy_disp_map.get((apt, str(rwy).strip()), np.nan) for apt, rwy in zip(df['ADEP_mvt'], df['RUNWAY_mvt'])], dtype=np.float32)
    return r_len, r_head, r_disp

r_len_tr, r_head_tr, r_disp_tr = get_rwy_features(df_train)
r_len_rk, r_head_rk, r_disp_rk = get_rwy_features(df_rank)

# -------------------------------------------------------------
# 2. CDM ÇAPRAZ ZAMANLAMA MATRİSİ (AOBT, EOBT, LOBT, IOBT, ARVT3, SCHED)
# -------------------------------------------------------------
print("\n[3/4] CDM Çapraz Zamanlama Fark Matrisi Hesaplanıyor...")
def get_cdm_deltas(df):
    aobt  = df['AOBT_DIFF_SEC'].fillna(0).astype(np.float32)
    eobt  = df['EOBT_DIFF_SEC'].fillna(0).astype(np.float32)
    lobt  = df['LOBT_DIFF_SEC'].fillna(0).astype(np.float32)
    iobt  = df['IOBT_DIFF_SEC'].fillna(0).astype(np.float32)
    arvt3 = df['ARVT3_DIFF_SEC'].fillna(0).astype(np.float32)
    sched = df['SCHED_DIFF_SEC'].fillna(0).astype(np.float32)
    
    return pd.DataFrame({
        'AOBT_MINUS_EOBT_v3': aobt - eobt,
        'AOBT_MINUS_LOBT_v3': aobt - lobt,
        'AOBT_MINUS_IOBT_v3': aobt - iobt,
        'EOBT_MINUS_IOBT_v3': eobt - iobt,
        'EOBT_MINUS_LOBT_v3': eobt - lobt,
        'ARVT3_MINUS_AOBT_v3': arvt3 - aobt,
        'SCHED_MINUS_AOBT_v3': sched - aobt
    })

cdm_tr = get_cdm_deltas(df_train)
cdm_rk = get_cdm_deltas(df_rank)

# -------------------------------------------------------------
# 3. STAND-RUNWAY VE AIRPORT HİYERARŞİK TARGET ENCODING
# -------------------------------------------------------------
print("\n[4/4] 5-Fold K-Katlı Sızıntısız Stand-Runway Target Encoding Hesaplanıyor...")
# Sadece Model A uçuşları için NM_OFFSET_SEC hedefi üzerinden
global_mean = df_train['NM_OFFSET_SEC'].dropna().mean()
m_weight = 15.0 # Bayesian smoothing weight

# (ADEP, RUNWAY, STAND) bileşik anahtarı
key_tr = df_train['ADEP_mvt'].astype(str) + "_" + df_train['RUNWAY_mvt'].astype(str) + "_" + df_train['STAND_mvt'].astype(str)
key_rk = df_rank['ADEP_mvt'].astype(str) + "_" + df_rank['RUNWAY_mvt'].astype(str) + "_" + df_rank['STAND_mvt'].astype(str)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
te_oof = np.full(len(df_train), np.nan, dtype=np.float32)
target = df_train['NM_OFFSET_SEC'].values

for tr_idx, val_idx in kf.split(df_train):
    sub_keys = key_tr.iloc[tr_idx]
    sub_targets = target[tr_idx]
    valid_mask = pd.notna(sub_targets)
    
    # Calculate smoothed means
    stats = pd.DataFrame({'k': sub_keys[valid_mask], 'y': sub_targets[valid_mask]}).groupby('k')['y'].agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + m_weight * global_mean) / (stats['count'] + m_weight)
    smooth_dict = smoothed.to_dict()
    
    val_keys = key_tr.iloc[val_idx]
    te_oof[val_idx] = val_keys.map(smooth_dict).fillna(global_mean).values.astype(np.float32)

# Test seti için tüm eğitim verisi üzerinden
valid_all = pd.notna(target)
stats_all = pd.DataFrame({'k': key_tr[valid_all], 'y': target[valid_all]}).groupby('k')['y'].agg(['count', 'mean'])
smoothed_all = (stats_all['count'] * stats_all['mean'] + m_weight * global_mean) / (stats_all['count'] + m_weight)
smooth_dict_all = smoothed_all.to_dict()
te_rank = key_rk.map(smooth_dict_all).fillna(global_mean).values.astype(np.float32)

# Birleştir ve Parquet olarak kaydet
feat_train_df = pd.DataFrame({
    'MVT_ID_mvt': df_train['MVT_ID_mvt'].values,
    'RWY_PHYSICAL_LENGTH_FT': r_len_tr,
    'RWY_HEADING_DEG_T': r_head_tr,
    'RWY_DISPLACED_THRESH_FT': r_disp_tr,
    'STAND_RWY_TE_OFFSET': te_oof
})
for c in cdm_tr.columns:
    feat_train_df[c] = cdm_tr[c].values

feat_rank_df = pd.DataFrame({
    'MVT_ID_mvt': df_rank['MVT_ID_mvt'].values,
    'RWY_PHYSICAL_LENGTH_FT': r_len_rk,
    'RWY_HEADING_DEG_T': r_head_rk,
    'RWY_DISPLACED_THRESH_FT': r_disp_rk,
    'STAND_RWY_TE_OFFSET': te_rank
})
for c in cdm_rk.columns:
    feat_rank_df[c] = cdm_rk[c].values

os.makedirs(os.path.dirname(OUT_FEAT_TRAIN), exist_ok=True)
feat_train_df.to_parquet(OUT_FEAT_TRAIN, index=False)
feat_rank_df.to_parquet(OUT_FEAT_RANK, index=False)

print("\n" + "=" * 100)
print(f"✅ Gelişmiş Özellik Dosyaları Başarıyla Oluşturuldu:")
print(f"  Eğitim Dosyası: {OUT_FEAT_TRAIN} ({os.path.getsize(OUT_FEAT_TRAIN)/1e6:.1f} MB)")
print(f"  Test Dosyası  : {OUT_FEAT_RANK} ({os.path.getsize(OUT_FEAT_RANK)/1e6:.1f} MB)")
print(f"  Toplam Süre   : {time.time() - t0:.1f} sn")
print("=" * 100)
