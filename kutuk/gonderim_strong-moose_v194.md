# Gönderim denetimi — strong-moose_v194

- Dosya: `<repo>\gonderim\strong-moose_v194.parquet`
- MD5: `388e9f787e9a209272c4a9629b7823ab`
- Aday: `D_kqgpx_adsb_16a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpx_16a_b5_d1 satırlarında yeni (5290 satır), diğerleri v126
- Not: v193 ile aynı; NM'siz satırlarda yalnız LTFM ve LFPG'de D1 kalıntı düzeltmesi (Model A'nın gecikme / pencere / METAR dosyalarıyla; CV, 24 saat hariç: LTFM 897 → 779, LFPG 1.083 → 1.065; LB beklentisi ≈ −1,3).

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
| model_a_rms_fark | 123.92 |
| model_a_ort_fark | -6.68 |
| yeni_modelin_v126dan_rms_farki | 123.92 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,772.90 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,772.90 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 464.59 |
| model_b_ort_fark | 20.41 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,666.39 |