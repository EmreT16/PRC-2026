# Gönderim denetimi — strong-moose_v167

- Dosya: `<repo>\gonderim\strong-moose_v167.parquet`
- MD5: `ccff5687747abc0eb4151d3e4a32819d`
- Aday: `D_dw_adsb_10d`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v164 + LIRF Model B rejim sınıflandırıcısına havayolu ve park yeri yer tutucu oranları (tüm LIRF kalkışlarından, ay dışı; çeyrek CV ön koşul dışı Q1 2.362->2.202, Q2 3.180->2.970, Q3 3.622->3.382, Q4 2.291->2.263). v166'nın yerine.

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
| model_a_rms_fark | 100.97 |
| model_a_ort_fark | -2.20 |
| yeni_modelin_v126dan_rms_farki | 100.97 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 348.09 |
| model_b_ort_fark | 14.99 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,660.97 |