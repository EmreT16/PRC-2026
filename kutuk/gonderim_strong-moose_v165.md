# Gönderim denetimi — strong-moose_v165

- Dosya: `<repo>\gonderim\strong-moose_v165.parquet`
- MD5: `4205938b17cba210e778662a077f005a`
- Aday: `D_dw_adsb_10d_k115`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb satırlarında yeni (5290 satır), diğerleri v126
- Not: v164 ile aynı düzeltici, k = 1,15 (düzeltme %15 büyütüldü). 2025 CV'de k=1,25 +0,4 kötüydü; ama v161/v163 LB kazancı CV'nin öngördüğünden büyük geldi (2026 kapsaması daha iyi). v164 ile farkı 2026'da daha güçlü düzeltmenin işe yarayıp yaramadığını ölçer.

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
| model_a_rms_fark | 113.86 |
| model_a_ort_fark | -2.65 |
| yeni_modelin_v126dan_rms_farki | 113.86 |
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