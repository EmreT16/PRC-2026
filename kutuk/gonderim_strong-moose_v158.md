# Gönderim denetimi — strong-moose_v158

- Dosya: `<repo>\gonderim\strong-moose_v158.parquet`
- MD5: `56184639913db423d4471f9053e8ba86`
- Aday: `D_dw_adsb_6c`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_bant_adsb satırlarında yeni (5290 satır), diğerleri v126
- Not: Model A: v149 (D_dw) + ADS-B düzeltici v2 (Huber LGB 3 tohum; Oca+Şub+Tem+Ağu 2025 ile eğitildi; yerel sapma + taban std özellikleri; EHAM/EDDF/LSZH/LEBL/EDDM/EGLL; Oca<->Tem CV 224,50 -> 212,52). Model B: v157 ile aynı (tum6M + bant + beş havalimanında ADS-B).

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
| model_a_rms_fark | 91.86 |
| model_a_ort_fark | -1.17 |
| yeni_modelin_v126dan_rms_farki | 91.86 |
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