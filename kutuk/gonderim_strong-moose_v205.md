# Gönderim denetimi — strong-moose_v205

- Dosya: `<repo>\gonderim\strong-moose_v205.parquet`
- MD5: `faf90dec4f1f14ca43e7431dd2508454`
- Aday: `D_kqgpT2_adsb_25a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v202 kurgusu (düzeltici OOF tarafı kqgpxg, 12 ay) + Model A test tabanı komşu 2. turla (K2, K2OFS, K2V, K2Q; scripts/komsu2_oz.py) yeniden eğitildi. v202 ile farkı = 2. turun etkisi (düzeltici değişmeden).

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
| model_a_rms_fark | 133.81 |
| model_a_ort_fark | -8.18 |
| yeni_modelin_v126dan_rms_farki | 133.81 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,660.43 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,660.43 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |