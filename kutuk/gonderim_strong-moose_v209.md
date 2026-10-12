# Gönderim denetimi — strong-moose_v209

- Dosya: `<repo>\gonderim\strong-moose_v209.parquet`
- MD5: `96d97dce25ddc29913f2c7eaf9997a5b`
- Aday: `D_kqgpT6_adsb_29a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v208 ile aynı (komşu + yankı geçmişi + günlük seriler) ama GCB (2 tohum) ve DW (3 tohum) test modelleri CV iç erken durdurmasına uygun iterasyonla: GCB 10.000 → 4.000, DW 9.000 → 3.500 (komşulu modellerde erken durdurma GCB 1.623–3.927, DW 2.433–3.688). v208 ile farkı = iterasyon düzeltmesinin etkisi.

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
| model_a_rms_fark | 132.84 |
| model_a_ort_fark | -8.99 |
| yeni_modelin_v126dan_rms_farki | 132.84 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,354.57 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,354.57 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |