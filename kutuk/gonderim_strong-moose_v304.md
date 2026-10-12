# Gönderim denetimi — strong-moose_v304

- Dosya: `<repo>\gonderim\strong-moose_v304.parquet`
- MD5: `4c540834b8a464eb35f229d56f09b875`
- Aday: `D_kqgpPe_adsb_42o_ha_g2lgbcb_g2t`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb satırlarında yeni (5290 satır), diğerleri v126
- Not: v301 ile aynı (B: 6 havalimanı) + G2b (G2 dışındaki satırlarda LightGBM + CatBoost ikinci aşama, PRC_G2_TERS=1; CV −%0,467, tam kalibrasyonla −%0,483). Ara dosya.

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
| model_a_rms_fark | 140.29 |
| model_a_ort_fark | -2.97 |
| yeni_modelin_v126dan_rms_farki | 140.29 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,188.09 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,188.09 |
| lobt_kirpilan_satir | 22.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 456.26 |
| model_b_ort_fark | 0.37 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,646.34 |