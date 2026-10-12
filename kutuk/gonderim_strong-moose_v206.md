# Gönderim denetimi — strong-moose_v206

- Dosya: `<repo>\gonderim\strong-moose_v206.parquet`
- MD5: `697e698aaf6c6d5e49bdcf8d1f086105`
- Aday: `D_kqgpT3_adsb_26a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v205 ile aynı (komşu 1. + 2. tur, düzeltici OOF kqgpxg) + Model A'ya yankı geçmişi özellikleri (gecikmeli kalkışta BLOCK = tarife oranı; uçuş kodu, havayolu, varış, park; ay dışı hedef kodlama, scripts/yanki_te_ozellik.py). v205 ile farkı = yankı geçmişinin etkisi.

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
| model_a_rms_fark | 133.58 |
| model_a_ort_fark | -7.38 |
| yeni_modelin_v126dan_rms_farki | 133.58 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,476.80 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,476.80 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |