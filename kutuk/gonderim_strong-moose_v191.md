# Gönderim denetimi — strong-moose_v191

- Dosya: `<repo>\gonderim\strong-moose_v191.parquet`
- MD5: `d2f3046cf14153b371db3b76d3cc0425`
- Aday: `D_kqgp_adsb_15c_k090`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgp_15a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v190 ile aynı, düzeltme 0,9 katı (k=0,9; LIRF yankı hedge'i korundu). v189 (k=1,15) +0,39 kötü çıktı; k'nın 1'in altında olup olmadığını ölçer.

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
| model_a_rms_fark | 115.97 |
| model_a_ort_fark | -4.34 |
| yeni_modelin_v126dan_rms_farki | 115.97 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,729.69 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,729.68 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 366.61 |
| model_b_ort_fark | -2.74 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,643.23 |