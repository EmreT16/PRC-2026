# Gönderim denetimi — strong-moose_v217

- Dosya: `<repo>\gonderim\strong-moose_v217.parquet`
- MD5: `56ed31110e0b46f9de96e7c4bbf70085`
- Aday: `D_kqgpT8b_adsb_34a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v211 ile aynı (T8 ağırlıkları, GCB 4 / DW 5 tohum) + lgbh3 / E1 6 tohum (1, 2, 3 eklendi) + LOBT. v211 ile farkı = LightGBM tohumları (küçük).

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
| model_a_rms_fark | 133.51 |
| model_a_ort_fark | -7.87 |
| yeni_modelin_v126dan_rms_farki | 133.27 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,779.51 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,779.51 |
| lobt_kirpilan_satir | 40.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |