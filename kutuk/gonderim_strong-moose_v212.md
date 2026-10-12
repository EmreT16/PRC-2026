# Gönderim denetimi — strong-moose_v212

- Dosya: `<repo>\gonderim\strong-moose_v212.parquet`
- MD5: `5d3d92707966bfdaee3d983346d009a8`
- Kaynak: `gonderim/strong-moose_v208.parquet` (MD5 95754c23f313d3310b17f5a655ce7a9b) + Y1 kalibrasyonu (scripts/g_kalibre.py) + LOBT penceresi
- Not: v208 + Y1 EUROCONTROL aylık taksi-out kalibrasyonu (scripts/g_kalibre.py; 2025 tanım farkı düşülür, 2·σ yumuşak eşik, geçersiz payı > %10 aylar atlanır) + LOBT penceresi. v208 ile farkı = Y1.

```
 apt  ay     n    TF  geçersiz_%  yayımlanan  fk_ort  fk_std  sapma  kayma
EDDF   1 15667 15667        12.7       853.5     0.5     2.8   -7.2    0.0
EDDF   7 20650 20650         0.0       854.2     0.5     2.8    8.1   -2.5
EDDM   1 10953 11033        31.1       755.0    11.4     2.1  -14.9    0.0
EDDM   7 15013 15131         0.8       776.3    11.4     2.1   -5.2    0.9
EGLL   1 19283 19283         0.0      1334.9     0.9     0.7  -11.8   10.5
EGLL   7 20557 20556         0.0      1344.1     0.9     0.7   -5.1    3.8
EHAM   1 16232 16321         1.1       982.1    10.8     4.9  -28.8   18.9
EHAM   7 21950 22060         0.9       840.9    10.8     4.9   -2.7    0.0
LEBL   1 12201 12201         0.0       883.1    -2.4     2.2    3.2   -0.0
LEBL   7 17879 17867         0.0       967.2    -2.4     2.2    8.6   -4.1
LEMD   1 16802 16787         3.1       987.3    -2.2     4.6   -6.1    0.0
LEMD   7 20152 20127         0.0      1020.4    -2.2     4.6   -5.2    0.0
LFPG   1 18129 18131         9.8       928.2   -18.9    17.4  -23.1    0.0
LFPG   7 21743 21743         0.0       965.1   -18.9    17.4  -34.0    0.0
LIRF   1 11061 11061         0.4      1043.3    -9.1     3.9  -10.9    3.1
LIRF   7 15838 15834         0.5      1214.0    -9.1     3.9   61.5  -53.6
LSZH   1  9995 10012        19.7       711.0     0.6     6.0   -6.9    0.0
LSZH   7 13155 13166         0.7       741.5     0.6     6.0    3.6   -0.0
LTFM   1 22396 22455         4.8      1088.7   -18.3     8.6   31.0  -13.9
LTFM   7 25185 25265         0.0      1008.8   -18.3     8.6   -3.1    0.0
```

- Değişim: RMS 13.58 sn, ort -1.86 sn; LOBT kırpılan 39