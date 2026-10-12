# Gönderim denetimi — strong-moose_v183

- Dosya: `<repo>\gonderim\strong-moose_v183.parquet`
- MD5: `bcb9fd7bdeb1217901ac8e9734837e58`
- Aday: `D_kqg_adsb_13b`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqg_13a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v181 ile aynı, düzeltici kapsama yoğunluğu özellikleri olmadan (ADSB_N, yerel_n, ADSB_BOSLUK, ADSB_BOSLUK5). v181 ile farkı alıcı ağı değişiminin etkisini ölçer.

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
| model_a_rms_fark | 116.97 |
| model_a_ort_fark | -3.12 |
| yeni_modelin_v126dan_rms_farki | 116.97 |
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