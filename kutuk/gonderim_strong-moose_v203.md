# Gönderim denetimi — strong-moose_v203

- Dosya: `<repo>\gonderim\strong-moose_v203.parquet`
- MD5: `d64d4de58ecb4c32d8d4e89d2b595796`
- Aday: `D_kqgpy_adsb_23a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v202 ile aynı test tabanı (Model A + komşu özellikleri KOM/KTAH) ama ADS-B düzelticisi komşulu Model A'nın 12 aylık OOF'larıyla eğitildi (harman kqgpy). Düzeltici CV Oca/Tem: kqgpxg 198.03 → kqgpy 199.59.

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
| model_a_rms_fark | 131.61 |
| model_a_ort_fark | -8.02 |
| yeni_modelin_v126dan_rms_farki | 131.61 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,408.63 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,408.63 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |