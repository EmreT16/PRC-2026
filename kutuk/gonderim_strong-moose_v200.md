# Gönderim denetimi — strong-moose_v200

- Dosya: `<repo>\gonderim\strong-moose_v200.parquet`
- MD5: `f28f1e2f36e593427459fca23bd9b517`
- Aday: `D_kqgpxg_adsb_20a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: kqgpxg (GCB 12.000; iz5, seyreltilmiş kopyalar, düzeltici δ=600 — v198 LB: δ=400 +0,05, D1 yok); ADS-B düzelticisi 8 yerine 12 ayla eğitildi (v199 + Nisan, Mayıs, Eylül, Ekim 2025; Model A split_5/6 OOF). CV Oca/Tem: 198.44 → 198.03.

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
| model_a_rms_fark | 124.76 |
| model_a_ort_fark | -6.11 |
| yeni_modelin_v126dan_rms_farki | 124.76 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,825.49 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,825.49 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |