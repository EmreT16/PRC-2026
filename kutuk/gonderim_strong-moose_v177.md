# Gönderim denetimi — strong-moose_v177

- Dosya: `<repo>\gonderim\strong-moose_v177.parquet`
- MD5: `0fa637026f57f3486ecf49281102d28e`
- Aday: `D_kq4_adsb_11a_k115`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kq4_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v176 ile aynı, düzeltme 1,15 katına büyütüldü (k=1,15; LIRF yankı hedge'i korundu). v176 ile farkı 2026'da daha güçlü düzeltmenin işe yarayıp yaramadığını ölçer (2025 CV'de k=1,25 +1,2 kötüydü; CV kazançları LB'ye büyüyerek taşındı).

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
| model_a_rms_fark | 127.22 |
| model_a_ort_fark | -1.46 |
| yeni_modelin_v126dan_rms_farki | 127.22 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,692.02 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,692.02 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 367.94 |
| model_b_ort_fark | -3.57 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.40 |