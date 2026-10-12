# Gönderim denetimi — strong-moose_v210

- Dosya: `<repo>\gonderim\strong-moose_v210.parquet`
- MD5: `ee4b0d60a7ef761c19f05c3fab2b3474`
- Aday: `D_kqgpT7_adsb_30a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v208 ile aynı (komşu + yankı geçmişi + günlük seriler; GCB 10.000 / DW 9.000 it.) ama test tarafında çok tohum: GCB 2 → 4 tohum (42, 2026, 7, 123), DW 3 → 5 tohum (42, 2026, 7, 123, 999); + LOBT penceresi (taksi [MVT − LOBT ± 3606] içine kırpılır; 2025'te kural %100). v208 ile farkı = tohum ortalaması (+ ~40 satırlık kırpma).

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
| model_a_rms_fark | 133.65 |
| model_a_ort_fark | -8.32 |
| yeni_modelin_v126dan_rms_farki | 133.46 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,313.53 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,313.53 |
| lobt_kirpilan_satir | 40.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |