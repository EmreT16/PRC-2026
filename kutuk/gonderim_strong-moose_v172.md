# Gönderim denetimi — strong-moose_v172

- Dosya: `<repo>\gonderim\strong-moose_v172.parquet`
- MD5: `9fa7bda69d88c85dd6b90ed9a3833872`
- Aday: `D_dw_adsb_10g_k115`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v170 ile aynı, düzeltme 1,15 katına büyütüldü (k=1,15; LIRF yankı hedge'i aynen korundu). 2025 CV'de k=1,25 +1,2 kötü, ama CV kazançları LB'ye büyüyerek taşındı (v161: CV −10 -> LB −16,8; v170: CV −3,3 -> LB −6,2); 2026 kapsaması daha iyi olduğu için güçlü düzeltme işe yarayabilir. v170 ile farkı bunu ölçer.

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
| model_a_rms_fark | 124.51 |
| model_a_ort_fark | -1.35 |
| yeni_modelin_v126dan_rms_farki | 124.51 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 348.09 |
| model_b_ort_fark | 14.99 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,660.97 |