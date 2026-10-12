# Gönderim denetimi — strong-moose_v154

- Dosya: `<repo>\gonderim\strong-moose_v154.parquet`
- MD5: `ba1a602647c4fe38621d6e29c5c8a9d3`
- Aday: `D_dw_adsb_5`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_bant satırlarında yeni (2981 satır), diğerleri v126
- Not: Model A: v149 (D_dw, dondurulan) + ADS-B düzeltici (adsb.lol, ODbL; Huber LightGBM, 3 tohum; Ocak+Temmuz 2025 ile eğitildi; yalnızca EHAM/EDDF/LSZH/LEBL/EDDM; Oca<->Tem CV 224,50 -> 214,32). Model B: v153 ile aynı (tum6M + bant).

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
| model_a_rms_fark | 82.45 |
| model_a_ort_fark | -0.96 |
| yeni_modelin_v126dan_rms_farki | 82.45 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 2,981.00 |
| model_b_rms_fark | 691.43 |
| model_b_ort_fark | 42.90 |
| model_b_ort_v126 | 1,754.12 |
| model_b_ort_yeni | 1,797.02 |