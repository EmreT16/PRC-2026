# Gönderim denetimi — strong-moose_v180

- Dosya: `<repo>\gonderim\strong-moose_v180.parquet`
- MD5: `51bb46bcb461548a1619787efbd38357`
- Aday: `D_kq4_adsb_12a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kq4_12a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v176 + ikinci aşama kalıntı modeli (kayan gecikme endeksi: son 15/30 dk'da kalkanların ortanca kalkış−EOBT gecikmesi, son 60 dk tarife gecikmesi, uçuşun kendi farkı), k=0,75, Model A tabanına ADS-B düzelticisinden önce eklendi. İkinci aşama 6 ay OOF: Oca 199,11→197,77, Tem 240,71→238,38, Haz 211,38→209,73, Ağu 211,13→209,91, Ara 196,17→194,55. Düzeltici Oca/Tem CV 205,73 → 204,83.

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
| model_a_rms_fark | 117.77 |
| model_a_ort_fark | -3.69 |
| yeni_modelin_v126dan_rms_farki | 117.77 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,935.53 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,935.53 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 371.69 |
| model_b_ort_fark | -6.55 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,639.43 |