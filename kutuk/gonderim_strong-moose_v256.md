# Gönderim denetimi — strong-moose_v256

- Dosya: `<repo>\gonderim\strong-moose_v256.parquet`
- MD5: `7128ba056417198e1d7574b0f96db5c1`
- Aday: `D_kqgpPe_adsb_40o`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz_d5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v243 ile aynı, Model A ADS-B düzelticisi _40o: OPDI özellikleri + OPDI'nin yerde gördüğü izsiz satırlar + mvt_epoch 12 ay (Y8). Ara dosya.

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
| model_a_rms_fark | 135.02 |
| model_a_ort_fark | -6.37 |
| yeni_modelin_v126dan_rms_farki | 134.83 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,418.34 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,418.34 |
| lobt_kirpilan_satir | 36.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 398.59 |
| model_b_ort_fark | 3.43 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,649.41 |