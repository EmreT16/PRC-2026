# Gönderim denetimi — strong-moose_v195

- Dosya: `<repo>\gonderim\strong-moose_v195.parquet`
- MD5: `d88089b289c3e1e2dc536f89dcf856fd`
- Aday: `D_kqgpxg_adsb_17a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5_d1 satırlarında yeni (5290 satır), diğerleri v126
- Not: v194 ile aynı (iz5, seyreltilmiş kopyalar, NM'siz LTFM/LFPG D1), Model A'da global CatBoost 12.000 iterasyon (kqgpxg).

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
| model_a_rms_fark | 124.01 |
| model_a_ort_fark | -6.60 |
| yeni_modelin_v126dan_rms_farki | 124.01 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,825.49 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,825.49 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 464.18 |
| model_b_ort_fark | 20.10 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,666.08 |