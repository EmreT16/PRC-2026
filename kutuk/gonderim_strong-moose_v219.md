# Gönderim denetimi — strong-moose_v219

- Dosya: `<repo>\gonderim\strong-moose_v219.parquet`
- MD5: `894424105aab0f80aed199263910fed7`
- Aday: `D_kqgpP_adsb_35a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: Pistli final yapısı: v215'in özellikleri (komşu + yankı + günlük + pist konfigürasyonu) + T8 ağırlıkları + çok tohum (lgbh3 / E1 6, GCB 4, DW 5; XGB 2) + LOBT.

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
| model_a_rms_fark | 133.29 |
| model_a_ort_fark | -8.14 |
| yeni_modelin_v126dan_rms_farki | 133.04 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,849.80 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,849.81 |
| lobt_kirpilan_satir | 42.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |