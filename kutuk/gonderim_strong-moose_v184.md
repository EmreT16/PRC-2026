# Gönderim denetimi — strong-moose_v184

- Dosya: `<repo>\gonderim\strong-moose_v184.parquet`
- MD5: `a83825f2d69e2a35d5fc3f670fe58b71`
- Aday: `D_kqgp_adsb_14a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgp_14a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: Model A tabanı: kuyruk + kayan gecikme endeksi + pist bazlı gecikme endeksi (GEC_PIST_*) + varış yoğunluğu (AR_*) + METAR hava bayrakları (WX_*; IEM, kamu malı) + ileriye bakan / ortalanmış pencereler (GECC_*, NMT_*, *_SONRA_10) + aynı park yerinde önceki / sonraki varış (IN_*, NX_*) 5 bileşene gömülü (kqgp; D_dw ağırlıkları). ADS-B: adsb4, düzeltici 6 ay, 255/800, kuyruk özellikleri, LIRF yankı hedge'i β=0,25. B satırları v5 düzeltici 8 havalimanı.

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
| model_a_rms_fark | 122.18 |
| model_a_ort_fark | -4.97 |
| yeni_modelin_v126dan_rms_farki | 122.18 |
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