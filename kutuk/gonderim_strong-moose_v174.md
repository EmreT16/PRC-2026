# Gönderim denetimi — strong-moose_v174

- Dosya: `<repo>\gonderim\strong-moose_v174.parquet`
- MD5: `247a5f890ff06e3dc7a15898566dd4e7`
- Aday: `D_dw_adsb_10i`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb5g_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v173 + üçüncü ADS-B eşleştirmesi (kalkış anı tespit edilemeyen ama aynı çağrı koduyla yerde görülen izler; data/adsb3). Test izli satır %69,2 -> %71,7 (Ocak EGLL %85->%93, EDDF %83->%88, LFPG %16->%27, LIRF %78->%86). Düzeltici Oca/Tem CV 207,64 -> 206,65 (Ocak -1,60, Temmuz -0,62; EGLL 224,6 -> 220,3). Model A tabanı v149 (D_dw); B satırları v5 düzeltici (adsb3 ile yeniden eğitildi).

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
| model_a_rms_fark | 112.50 |
| model_a_ort_fark | -0.89 |
| yeni_modelin_v126dan_rms_farki | 112.50 |
| 40000_ustu | 15.00 |
| 40000_ustu_v126 | 15.00 |
| satir_24s | 85,927.64 |
| satir_24s_v126 | 86,074.27 |
| satir_24s_yeni | 85,927.64 |
| model_b_degisen_satir | 5,290.00 |
| model_b_rms_fark | 356.92 |
| model_b_ort_fark | -6.57 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,639.41 |