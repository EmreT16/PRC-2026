# Gönderim denetimi — strong-moose_v161

- Dosya: `<repo>\gonderim\strong-moose_v161.parquet`
- MD5: `02d8dea4c155a7216a6409c374021dde`
- Aday: `D_dw_adsb_6c`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb satırlarında yeni (5290 satır), diğerleri v126
- Not: v158'in bant kuralsız hâli (bant kuralı v153'te LB'de +0,74 kaybettirdi). Model A: v149 + ADS-B düzeltici v2, 6 havalimanı (EHAM/EDDF/LSZH/LEBL/EDDM/EGLL; Oca<->Tem CV 224,50 -> 212,52). Model B: v152 + beş havalimanında ADS-B (1.709 satır). v152 ile farkı ADS-B'nin toplam etkisini ölçer.

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
| model_b_rms_fark | 281.50 |
| model_b_ort_fark | 20.58 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,666.56 |