# Gönderim denetimi — strong-moose_v160

- Dosya: `<repo>\gonderim\strong-moose_v160.parquet`
- MD5: `447d83d672f42b2ea5c86637d0dabdbe`
- Aday: `D_dw_adsb_5c`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_bant_adsb satırlarında yeni (5290 satır), diğerleri v126
- Not: v158'in EGLL'siz hâli (ADS-B düzeltici v2 yalnızca EHAM/EDDF/LSZH/LEBL/EDDM; Oca<->Tem CV 224,50 -> 213,17). v158 ile farkı EGLL'in LB etkisini ölçer. Model B: v158 ile aynı.

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
| model_b_rms_fark | 529.98 |
| model_b_ort_fark | 31.24 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,677.22 |