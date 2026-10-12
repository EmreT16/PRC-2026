# Gönderim denetimi — strong-moose_v168

- Dosya: `<repo>\gonderim\strong-moose_v168.parquet`
- MD5: `b61c52c35870297c39a083d2b2a9e8cb`
- Aday: `D_dw_adsb_10e`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v167 + yeni ADS-B düzeltici (v4 özellikleri): park yeri koordinatlarından uzaklıklar, kayıt alışkanlığı kodlaması (havalimanı×havayolu, havalimanı×park yeri; gerçek taksi − ADS-B hareket), uçak tipi. Oca<->Tem CV 210,95 -> 208,57, iki ayda da iyi. Temmuz 2026 günleri C: üzerinde yeniden işlendi (konum sütunları). Model B: v167 ile aynı.

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
| model_a_rms_fark | 106.63 |
| model_a_ort_fark | -1.16 |
| yeni_modelin_v126dan_rms_farki | 106.63 |
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