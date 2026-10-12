# Gönderim denetimi — strong-moose_v156

- Dosya: `<repo>\gonderim\strong-moose_v156.parquet`
- MD5: `062ffa1ff2a136b274b5c495e65f60d9`
- Aday: `D_dw_adsb_8`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_bant satırlarında yeni (2981 satır), diğerleri v126
- Not: v155 + ADS-B düzeltici LEMD ve LIRF'te de (2025 CV'de LEMD ≈ 0, LIRF yalnız Temmuz -8,5; LFPG hariç). Model B: v153 ile aynı.

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
| model_a_rms_fark | 99.16 |
| model_a_ort_fark | -5.70 |
| yeni_modelin_v126dan_rms_farki | 99.16 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 2,981.00 |
| model_b_rms_fark | 691.43 |
| model_b_ort_fark | 42.90 |
| model_b_ort_v126 | 1,754.12 |
| model_b_ort_yeni | 1,797.02 |