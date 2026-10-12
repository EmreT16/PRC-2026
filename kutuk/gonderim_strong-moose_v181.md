# Gönderim denetimi — strong-moose_v181

- Dosya: `<repo>\gonderim\strong-moose_v181.parquet`
- MD5: `ad33c1499783bbe71b0f5e9e088794cf`
- Aday: `D_kqg_adsb_13a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqg_13a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: Model A tabanı: kuyruk + kayan gecikme endeksi özellikleri 5 bileşene gömülü (kqg: xgb, gcb6k_ort2, e1h3, lgbh3, DW; D_dw ağırlıkları). Bileşen düzeyinde kuyruk özellikliye göre ayda −1,5 … −6,4 sn. ADS-B: üçüncü eşleştirme + NM'siz satırlarda uçuş numarası (adsb4); düzeltici 6 ay, 255/800, kuyruk özellikleri, LIRF yankı hedge'i β=0,25. B satırları v5 düzeltici 8 havalimanı. İkinci aşama yok (endeks modelin içinde).

| Denetim | Sonuç |
|---|---|
| satir_sayisi | ✓ |
| sutunlar | ✓ |
| kimlik_sirasi | ✓ |
| nan_inf_yok | ✓ |
| alt_sinir_60 | ✓ |
| model_b_kalan_ayni | ✓ |
| model_b_yeni_uygulandi | ✓ |

| Bilgi | Değer |
|---|---:|
| model_a_rms_fark | 117.60 |
| model_a_ort_fark | -3.52 |
| yeni_modelin_v126dan_rms_farki | 117.60 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,675.28 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,675.28 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 366.84 |
| model_b_ort_fark | -2.89 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,643.09 |