# Gönderim denetimi — strong-moose_v173

- Dosya: `<repo>\gonderim\strong-moose_v173.parquet`
- MD5: `cd6b2cc48165e89de6a163a0166beda2`
- Aday: `D_dw_adsb_10g`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb5_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v170 + Model B (NM'siz) satırlarına v5 ADS-B düzeltici (6 ay, park/kayıt alışkanlığı/uçak tipi, 255 yaprak / 800 tur), 7 havalimanı (EDDF, EDDM, EHAM, LEBL, LSZH, EGLL, LEMD). 6 aylık CV (izli B satırları): EDDF 319->271, EDDM 540->442, EHAM 397->351, LEBL 297->253, LSZH 575->531, LEMD 279->250, EGLL 1019->1016; A5'te 6/6 ay iyi. LIRF/LFPG hariç (kötü/karışık). 2.055 satır, ort. −40,8 sn. Model A: v170 ile aynı.

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
| model_a_rms_fark | 110.05 |
| model_a_ort_fark | -1.04 |
| yeni_modelin_v126dan_rms_farki | 110.05 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 352.60 |
| model_b_ort_fark | -11.65 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,634.33 |