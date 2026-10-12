# Gönderim denetimi — strong-moose_v213

- Dosya: `<repo>\gonderim\strong-moose_v213.parquet`
- MD5: `76835ea5c6cce5ab8b09be38a0295b5d`
- Kaynak: `gonderim/strong-moose_v211.parquet` (MD5 0fdeedd7cb520333e6e20fdfa35fa98c) + Y1 kalibrasyonu (scripts/g_kalibre.py) + LOBT penceresi
- Not: strong-moose_v211.parquet + Y1 kalibrasyonu (aynı kural) + LOBT penceresi. Kaynakla farkı = Y1.

```
 apt  ay     n    TF  geçersiz_%  yayımlanan  fk_ort  fk_std  sapma  kayma
EDDF   1 15667 15667        12.7       853.5     0.5     2.8   -6.9    0.0
EDDF   7 20650 20650         0.0       854.2     0.5     2.8    8.7   -3.1
EDDM   1 10953 11033        31.1       755.0    11.4     2.1  -14.7    0.0
EDDM   7 15013 15131         0.8       776.3    11.4     2.1   -4.9    0.7
EGLL   1 19283 19283         0.0      1334.9     0.9     0.7  -11.3    9.9
EGLL   7 20557 20556         0.0      1344.1     0.9     0.7   -4.8    3.4
EHAM   1 16232 16321         1.1       982.1    10.8     4.9  -29.1   19.3
EHAM   7 21950 22060         0.9       840.9    10.8     4.9   -2.3    0.0
LEBL   1 12201 12201         0.0       883.1    -2.4     2.2    3.4   -0.0
LEBL   7 17879 17867         0.0       967.2    -2.4     2.2    8.9   -4.5
LEMD   1 16802 16787         3.1       987.3    -2.2     4.6   -5.6    0.0
LEMD   7 20152 20127         0.0      1020.4    -2.2     4.6   -5.0    0.0
LFPG   1 18129 18131         9.8       928.2   -18.9    17.4  -22.8    0.0
LFPG   7 21743 21743         0.0       965.1   -18.9    17.4  -32.5    0.0
LIRF   1 11061 11061         0.4      1043.3    -9.1     3.9   -9.7    1.8
LIRF   7 15838 15834         0.5      1214.0    -9.1     3.9   63.5  -55.6
LSZH   1  9995 10012        19.7       711.0     0.6     6.0   -6.5    0.0
LSZH   7 13155 13166         0.7       741.5     0.6     6.0    3.8   -0.0
LTFM   1 22396 22455         4.8      1088.7   -18.3     8.6   31.1  -14.0
LTFM   7 25185 25265         0.0      1008.8   -18.3     8.6   -3.2    0.0
```

- Değişim: RMS 13.42 sn, ort -2.13 sn; LOBT kırpılan 9