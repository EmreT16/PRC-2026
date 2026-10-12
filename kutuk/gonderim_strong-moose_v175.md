# Gönderim denetimi — strong-moose_v175

- Dosya: `<repo>\gonderim\strong-moose_v175.parquet`
- MD5: `0af718edd8fbbed9d9b505ff04758c40`
- Aday: `D_dw_adsb_10i`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_tum6M_adsb5h_lirfte satırlarında yeni (5290 satır), diğerleri v126
- Not: v174 + NM'siz (Model B) satırlarda uçuş numarası çağrı kodu olarak (ICAO; IATA önekleri veriden çıkarılan tabloyla çevrildi, data/iata_icao.json) → NM'siz satırlarda ADS-B eşleşmesi testte %44 -> %57 (EGLL 79->97, LEBL 68->99, LEMD 62->97, EDDM 77->97). v5 B düzeltici 8 havalimanında (LFPG eklendi; 6 ay CV'de EDDF/EDDM/EGLL/EHAM/LEBL/LSZH 6/6 ay iyi, LEMD 4/5, LFPG 4/6). 2.898 B satırı. Model A: v174 ile aynı.

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
| model_b_rms_fark | 368.22 |
| model_b_ort_fark | -5.29 |
| model_b_ort_v126 | 1,645.98 |
| model_b_ort_yeni | 1,640.69 |