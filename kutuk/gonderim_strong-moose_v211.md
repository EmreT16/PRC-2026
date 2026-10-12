# Gönderim denetimi — strong-moose_v211

- Dosya: `<repo>\gonderim\strong-moose_v211.parquet`
- MD5: `0fdeedd7cb520333e6e20fdfa35fa98c`
- Aday: `D_kqgpT8_adsb_31a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v210 ile aynı (v208 + çok tohum GCB 4 / DW 5 + LOBT penceresi) ama Model A harman ağırlıkları 6 bölmede yeniden seçildi (scripts/harman_agirlik6.py; kqgpy ve kqgp OOF seçimlerinin ortalaması: xgb 0,183 / gcb 0,183 / e1h3 0,280 / lgbh3 0,100 / dw 0,254; eski 0,102 / 0,204 / 0,221 / 0,124 / 0,350; iç içe CV kqgpy −0,17 5/6, kqgp −0,05 6/6). Düzeltici yeni ağırlıklı OOF tabanıyla. v210 ile farkı = ağırlıklar.

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
| model_a_rms_fark | 133.55 |
| model_a_ort_fark | -7.89 |
| yeni_modelin_v126dan_rms_farki | 133.31 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,786.04 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,786.04 |
| lobt_kirpilan_satir | 40.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |