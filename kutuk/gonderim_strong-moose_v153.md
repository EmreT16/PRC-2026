# Gönderim denetimi — strong-moose_v153

- Dosya: `<repo>\gonderim\strong-moose_v153.parquet`
- MD5: `f6d2b5eee71dfe13a9ad0294aa7bf72c`
- Aday: `D_dw`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_bant satırlarında yeni (2981 satır), diğerleri v126
- Not: Model A: v149 ile aynı (dondurulan). Model B: v152 + LIRF 24 saat ön koşulu, gecikme 15,5–24 sa bandındaki 3 satır: p·87.575 + (1−p)·gecikme, p = 0,667 (eğitim 9/13, Laplace)

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
| model_a_rms_fark | 41.07 |
| model_a_ort_fark | 0.82 |
| yeni_modelin_v126dan_rms_farki | 41.07 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 2,981.00 |
| model_b_rms_fark | 691.43 |
| model_b_ort_fark | 42.90 |
| model_b_ort_v126 | 1,754.12 |
| model_b_ort_yeni | 1,797.02 |