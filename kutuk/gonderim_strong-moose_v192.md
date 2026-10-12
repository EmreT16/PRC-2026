# Gönderim denetimi — strong-moose_v192

- Dosya: `<repo>\gonderim\strong-moose_v192.parquet`
- MD5: `b875352dc732a7a82d1dec498de7a12b`
- Aday: `D_kqgp_adsb_15d`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgp_15d_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v188 ile aynı (kqgp tabanı, adsb5, iz5); düzeltici ayrıca seyreltilmiş iz kopyalarıyla eğitildi (yerdeki ADS-B noktalarının %30 ve %10'u; düzelticiyi 2026'daki yoğunluk kaymasına dayanıklı kılmak için). CV 198,91 → 198,77; asıl etki yoğunluk kaymasında (CV göremez).

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
| model_a_rms_fark | 123.68 |
| model_a_ort_fark | -6.91 |
| yeni_modelin_v126dan_rms_farki | 123.68 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,729.69 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,729.68 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 366.61 |
| model_b_ort_fark | -2.74 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,643.23 |