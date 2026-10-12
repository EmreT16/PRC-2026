# Gönderim denetimi — strong-moose_v157

- Dosya: `<repo>\gonderim\strong-moose_v157.parquet`
- MD5: `51ca82c0d345a6738638402591c0fdc3`
- Aday: `D_dw_adsb_5`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_bant_adsb satırlarında yeni (5290 satır), diğerleri v126
- Not: v154 + Model B satırlarında ADS-B düzeltici (Model A düzelticisi aynen, nm yok; EHAM/EDDF/LSZH/LEBL/EDDM, 1.709 satır; Oca<->Tem CV B satırları Ocak 733,7->714,4, Temmuz 407,8->396,4).

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
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 529.98 |
| model_b_ort_fark | 31.24 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,677.22 |