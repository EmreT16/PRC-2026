# Gönderim denetimi — strong-moose_v170

- Dosya: `<repo>\gonderim\strong-moose_v170.parquet`
- MD5: `3c73cd9d5685008008842f4c5e314c36`
- Aday: `D_dw_adsb_10g`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v168 + düzeltici 6 ayla eğitildi (Haziran+Aralık 2025 eklendi; split_3 Model A OOF'u), 255 yaprak / 800 tur, LIRF'te tarife yankısı hedge'i (β=0,25 · p_kilit; Huber düzeltici bu hedge'i siliyordu). Oca/Tem CV 208,45 -> 207,64; yankı hedge'i 6 aylık OOF'ta LIRF'in 4 ayında da iyi. Kapsama özellikleri dahil (v168 gibi). Model B: v167 ile aynı.

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
| model_a_rms_fark | 110.05 |
| model_a_ort_fark | -1.04 |
| yeni_modelin_v126dan_rms_farki | 110.05 |
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