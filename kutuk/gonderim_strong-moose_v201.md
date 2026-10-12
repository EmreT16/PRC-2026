# Gönderim denetimi — strong-moose_v201

- Dosya: `<repo>\gonderim\strong-moose_v201.parquet`
- MD5: `d2f58b9853a4fa3ed5dc9093b2cbb606`
- Aday: `D_kqgpT_adsb_21a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kqgpxg_17a_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: v200 ile aynı (12 aylık düzeltici, OOF tarafı kqgpxg) ama Model A'nın 5 bileşeninin TEST tahminleri sıcaklık / çiy noktası özellikleriyle (SIC_0, SIC_MIN12, SPREAD_0; IEM METAR) yeniden eğitildi. v200 ile farkı = sıcaklığın LB etkisi (ay dışı CV bu özelliği ölçemiyor).

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
| model_a_rms_fark | 124.79 |
| model_a_ort_fark | -5.92 |
| yeni_modelin_v126dan_rms_farki | 124.79 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,910.65 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,910.65 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 365.28 |
| model_b_ort_fark | -3.07 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,642.90 |