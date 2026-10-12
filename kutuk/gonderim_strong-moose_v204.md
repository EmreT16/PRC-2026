# Gönderim denetimi — strong-moose_v204

- Dosya: `<repo>\gonderim\strong-moose_v204.parquet`
- MD5: `940c76ced1dc26e9bf3f66ca13a9365c`
- Aday: `D_kqgpy2_adsb_24a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v203 ile aynı (düzeltici komşulu OOF, harman kqgpy) ama Model A'nın 5 bileşeninin TEST tahminleri komşu özelliklerinin 2. turuyla (K2,K2OFS,K2V,K2Q; komşu tahminleri komşulu tabandan, scripts/komsu2_oz.py) yeniden eğitildi. v203 ile farkı = 2. turun LB etkisi.

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
| model_a_rms_fark | 133.93 |
| model_a_ort_fark | -8.02 |
| yeni_modelin_v126dan_rms_farki | 133.93 |
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