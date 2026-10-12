# Gönderim denetimi — strong-moose_v179

- Dosya: `<repo>\gonderim\strong-moose_v179.parquet`
- MD5: `a948868873d1363fb64ca0f7177630f2`
- Aday: `D_kq_adsb_11c`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kq_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: Model A tabanı: tam kuyruk / pencere özellikli harman (kq: xgb, gcb6k_ort2, e1h3, lgbh3, DW; beşi de kuyruk özellikli; D_dw ağırlıkları). ADS-B: üçüncü eşleştirme + NM'siz satırlarda uçuş numarası (adsb4); düzeltici 6 ay, 255/800, kuyruk özellikleri, LIRF yankı hedge'i β=0,25. B satırları v5 düzeltici 8 havalimanı. v176'dan farkı: global CatBoost da kuyruk özellikli.

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
| model_a_rms_fark | 113.15 |
| model_a_ort_fark | -1.21 |
| yeni_modelin_v126dan_rms_farki | 113.15 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,594.28 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,594.28 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 367.88 |
| model_b_ort_fark | -4.17 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,641.81 |