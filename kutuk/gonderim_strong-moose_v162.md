# Gönderim denetimi — strong-moose_v162

- Dosya: `<repo>\gonderim\strong-moose_v162.parquet`
- MD5: `a1e573363f4817506116610d0b545c07`
- Aday: `D_dw_adsb_5c`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb satırlarında yeni (5290 satır), diğerleri v126
- Not: v161'in EGLL'siz hâli (v160'ın bant kuralsızı; CV 213,17). v161 ile farkı EGLL'in LB etkisini ölçer.

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
| model_a_rms_fark | 81.72 |
| model_a_ort_fark | 0.66 |
| yeni_modelin_v126dan_rms_farki | 81.72 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 281.50 |
| model_b_ort_fark | 20.58 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,666.56 |