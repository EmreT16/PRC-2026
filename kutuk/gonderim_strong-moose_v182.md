# Gönderim denetimi — strong-moose_v182

- Dosya: `<repo>\gonderim\strong-moose_v182.parquet`
- MD5: `0d57f765439385b5535555b21e306df1`
- Aday: `D_kqg_adsb_13a_k115`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqg_13a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v181 ile aynı, düzeltme 1,15 katı (k=1,15; LIRF yankı hedge'i korundu). v181 ile farkı güçlü düzeltmenin 2026 etkisini ölçer.

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
| model_a_rms_fark | 130.98 |
| model_a_ort_fark | -3.91 |
| yeni_modelin_v126dan_rms_farki | 130.98 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,675.28 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,675.28 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 366.84 |
| model_b_ort_fark | -2.89 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,643.09 |