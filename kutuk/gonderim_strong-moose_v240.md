# Gönderim denetimi — strong-moose_v240

- Dosya: `<repo>\gonderim\strong-moose_v240.parquet`
- MD5: `979b9b3ffa354393e1255674d3684500`
- Aday: `D_kqgpGe_adsb_40a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v223'ün MVT_DAY'siz karşılığı: test tarafındaki 5 bileşen final yapılandırmasıyla (ekoz_secim_final.json, aynı tohumlar) MVT_DAY olmadan; ağırlık, düzeltici, B, hedge, LOBT aynı. Ara dosya; kalibre edilmiş hali gönderilir.

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
| model_a_rms_fark | 133.59 |
| model_a_ort_fark | -8.73 |
| yeni_modelin_v126dan_rms_farki | 133.40 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,313.03 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,313.03 |
| lobt_kirpilan_satir | 46.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |