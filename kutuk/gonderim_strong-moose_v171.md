# Gönderim denetimi — strong-moose_v171

- Dosya: `<repo>\gonderim\strong-moose_v171.parquet`
- MD5: `4b251eac861f2fca65eb41efff8b0fe8`
- Aday: `D_dw_adsb_10h`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v170 ile aynı (6 ay, 255 yaprak / 800 tur, LIRF yankı hedge'i β=0,25), kapsama yoğunluğu özellikleri olmadan (ADSB_N, yerel_n, ADSB_BOSLUK, ADSB_BOSLUK5). Oca/Tem CV 207,79 (v170: 207,64). Yalnızca v169 LB'de v168'i geçerse v170 yerine yüklenmeli. Model B: v167 ile aynı.

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
| model_a_rms_fark | 109.25 |
| model_a_ort_fark | -0.35 |
| yeni_modelin_v126dan_rms_farki | 109.25 |
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