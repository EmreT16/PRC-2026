# Gönderim denetimi — strong-moose_v207

- Dosya: `<repo>\gonderim\strong-moose_v207.parquet`
- MD5: `559237b8d13125911a2b0bfb5cd2f00a`
- Aday: `D_kqgpT4_adsb_27a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v205 ile aynı (komşu 1. + 2. tur, düzeltici OOF kqgpxg) + Model A'ya EUROCONTROL günlük serileri (gün × havalimanı: slotlu kalkış payı, slot dışı geç / erken pay, kalkış öncesi gecikme tümü / ATC; scripts/gunluk_seri_ozellik.py). Yankı geçmişi yok. v205 ile farkı = günlük serilerin etkisi.

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
| model_a_rms_fark | 133.76 |
| model_a_ort_fark | -9.19 |
| yeni_modelin_v126dan_rms_farki | 133.76 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,560.20 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,560.20 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |