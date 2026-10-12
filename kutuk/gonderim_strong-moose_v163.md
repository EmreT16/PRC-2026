# Gönderim denetimi — strong-moose_v163

- Dosya: `<repo>\gonderim\strong-moose_v163.parquet`
- MD5: `df93686fbb6497bab065d73acd439823`
- Aday: `D_dw_adsb_10c`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb satırlarında yeni (5290 satır), diğerleri v126
- Not: v161 + ADS-B düzeltici v2 tüm havalimanlarında (v159'un bant kuralsızı; CV 211,48).

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
| model_a_rms_fark | 98.00 |
| model_a_ort_fark | -2.93 |
| yeni_modelin_v126dan_rms_farki | 98.00 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 281.50 |
| model_b_ort_fark | 20.58 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,666.56 |