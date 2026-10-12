# Gönderim denetimi — strong-moose_v186

- Dosya: `<repo>\gonderim\strong-moose_v186.parquet`
- MD5: `4b60dab6e139c0c066833e0c5c281d2f`
- Aday: `D_kqgp_adsb_14a_k115`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgp_14a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v184 ile aynı, düzeltme 1,15 katı (k=1,15; LIRF yankı hedge'i korundu). v184 ile farkı güçlü düzeltmenin 2026 etkisini ölçer.

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
| model_a_rms_fark | 134.57 |
| model_a_ort_fark | -5.29 |
| yeni_modelin_v126dan_rms_farki | 134.57 |
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