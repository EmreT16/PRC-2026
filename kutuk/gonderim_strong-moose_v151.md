# Gönderim denetimi — strong-moose_v151

- Dosya: `<repo>\gonderim\strong-moose_v151.parquet`
- MD5: `4b84772d70a851196abff1906cf23191`
- Aday: `D_dw`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6 satırlarında yeni (2621 satır), diğerleri v126
- Not: Model A: v149 ile aynı (dondurulan). Model B: EHAM, LSZH, EDDF ve LIRF v126; EDDM, EGLL, LEBL, LEMD, LFPG, LTFM tüm satırlarla eğitilen model (test-eğitim gecikme kayması olmayan, CV kazançlı havalimanları)

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
| model_b_degisen_satir | 2,621.00 |
| model_b_rms_fark | 332.02 |
| model_b_ort_fark | 3.07 |
| model_b_ort_v126 | 1,227.20 |
| model_b_ort_yeni | 1,230.27 |