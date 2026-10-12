# Gönderim denetimi — strong-moose_v176

- Dosya: `<repo>\gonderim\strong-moose_v176.parquet`
- MD5: `5aede0ece945043cbe1de8533af9fa78`
- Aday: `D_kq4_adsb_11a`, yeni modelin payı: 1.00; taban v126 (MD5 486e03b2ef36eb9cc9fe559fa12ff0ff)
- Model B: TEST_B_D_kq4_b5 satırlarında yeni (5290 satır), diğerleri v126
- Not: Model A tabanı: kuyruk / pencere özellikli ara harman (kq4: xgb, e1h3, lgbh3, DW kuyruk özellikli; GCB v149; D_dw ağırlıkları). 6 ay OOF taban: Oca −1,32, Şub −0,96, Haz −1,28, Tem −1,21, Ağu −1,32, Ara −1,40. ADS-B: üçüncü eşleştirme + NM'siz satırlarda uçuş numarası (adsb4); düzeltici 6 ay, 255/800, kuyruk özellikleri, LIRF yankı hedge'i β=0,25 (kq E1 p_kilit). Düzeltici Oca/Tem CV 205,73 (v170: 207,64). B satırları v5 düzeltici 8 havalimanı.

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
| model_a_rms_fark | 112.78 |
| model_a_ort_fark | -1.10 |
| yeni_modelin_v126dan_rms_farki | 112.78 |
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