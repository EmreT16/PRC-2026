# Gönderim denetimi — strong-moose_v178

- Dosya: `<repo>\gonderim\strong-moose_v178.parquet`
- MD5: `1fe060e11469a444ed1944cb24b2b519`
- Aday: `D_kq4_adsb_11b`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kq4_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v176 ile aynı, düzeltici kapsama yoğunluğu özellikleri olmadan (ADSB_N, yerel_n, ADSB_BOSLUK, ADSB_BOSLUK5). v176 ile farkı 2025→2026 alıcı ağı değişiminin etkisini ölçer.

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
| model_a_rms_fark | 112.25 |
| model_a_ort_fark | -0.64 |
| yeni_modelin_v126dan_rms_farki | 112.25 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,692.02 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,692.02 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 367.94 |
| model_b_ort_fark | -3.57 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.40 |