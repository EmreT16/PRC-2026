# Gönderim denetimi — strong-moose_v272

- Dosya: `<repo>\gonderim\strong-moose_v272.parquet`
- MD5: `716686b3b6a525f9c4f758495f45e236`
- Aday: `D_kqgpPe_adsb_40o_ha_g2`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi satırlarında yeni (5290 satır), diğerleri v126
- Not: v258 ile aynı + H1 (havalimanı ağırlıkları, scripts/havalimani_agirlik.py) + G2 (H1 tabanıyla eğitilmiş; scripts/gecikme_asama2.py ha=1). CV ay dışarıda SSE -%0,258. Ara dosya.

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
| model_a_rms_fark | 136.15 |
| model_a_ort_fark | -5.55 |
| yeni_modelin_v126dan_rms_farki | 135.99 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,227.00 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,227.00 |
| lobt_kirpilan_satir | 26.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 440.92 |
| model_b_ort_fark | 7.86 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,653.84 |