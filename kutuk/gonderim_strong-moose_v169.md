# Gönderim denetimi — strong-moose_v169

- Dosya: `<repo>\gonderim\strong-moose_v169.parquet`
- MD5: `a4f26307e7271c10fc7dd70c0149d0a0`
- Aday: `D_dw_adsb_10f`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v168 ile aynı düzeltici, kapsama yoğunluğu özellikleri olmadan (ADSB_N, yerel_n, ADSB_BOSLUK, ADSB_BOSLUK5 atıldı). Alıcı ağı 2025->2026 değişti, 'nerede duyuldu' özellikleri taşınmadı. 2025->2026 kayma: yerel_n +0,50 sd, ADSB_N +0,38 sd. CV 208,57 -> 208,69 (ihmal edilir). v168 ile LB farkı taşınma sorununu ölçer. Model B: v167 ile aynı.

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
| model_a_rms_fark | 106.55 |
| model_a_ort_fark | -1.15 |
| yeni_modelin_v126dan_rms_farki | 106.55 |
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