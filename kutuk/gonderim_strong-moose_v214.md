# Gönderim denetimi — strong-moose_v214

- Dosya: `<repo>\gonderim\strong-moose_v214.parquet`
- MD5: `b4030e306e2fdb4dc010076e6cf5c6b3`
- Aday: `D_kqgpT5_adsb_28a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5_d1_LTFM1_LFPG0 satırlarında yeni (5290 satır), diğerleri v126
- Not: v208 + YEDEK Y2b: NM'siz LTFM satırlarına D1 kalıntı düzeltmesi tam ölçekle (k = 1; 721 satır, ort. +60,5 sn), LFPG'de uygulanmadı (2026 düzeltmesi 2025 dağılımının dışında: ort. +7 → +108 sn). Kural eğitim verisinden (2025 ay dışı en iyi k LTFM 1,70–2,85; 1'de kesildi), LB'den değil. + LOBT penceresi. v208 ile farkı = Y2b.

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
| model_a_rms_fark | 133.73 |
| model_a_ort_fark | -8.39 |
| yeni_modelin_v126dan_rms_farki | 133.55 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,355.19 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,355.19 |
| lobt_kirpilan_satir | 40.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 395.36 |
| model_b_ort_fark | 5.17 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,651.15 |