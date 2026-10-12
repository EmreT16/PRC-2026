# Gönderim denetimi — strong-moose_v215

- Dosya: `<repo>\gonderim\strong-moose_v215.parquet`
- MD5: `f589c88bddc8fe6549f4c1c1d2db9bbf`
- Aday: `D_kqgpT9_adsb_33a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v208 ile aynı (komşu + yankı + günlük seriler; tohum ve ağırlıklar v208'inki) + Model A'ya pist konfigürasyonu özellikleri (scripts/pist_ozellik.py, PC_*: pist sayıları, konfigürasyon değişimi, pist payı / baskınlık, iz türbülanslı pist kuyruğu, ağır uçak, son 20 kalkış süresi, konfigürasyon TE) + LOBT penceresi. Taramada nötrdü (+0,003); durum özellikleri LB'de CV'den fazla kazandırdığı için LB sınaması. v208 ile farkı = pist (+ ~40 satırlık LOBT).

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
| model_a_rms_fark | 133.60 |
| model_a_ort_fark | -8.54 |
| yeni_modelin_v126dan_rms_farki | 133.40 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 86,442.82 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 86,442.82 |
| lobt_kirpilan_satir | 44.00 |
| lobt_kirpilan_model_b_satir | 0.00 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |