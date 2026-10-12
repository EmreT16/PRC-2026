# Gönderim denetimi — strong-moose_v225

- Dosya: `<repo>\gonderim\strong-moose_v225.parquet`
- MD5: `5ca7024eac5eb643586306a0820cdc07`
- Kaynak: `gonderim/strong-moose_v221.parquet` (MD5 279059f752b3968e65c950b87d44aa89) + Y1 kalibrasyonu (scripts/g_kalibre.py) + LOBT penceresi
- Not: strong-moose_v221.parquet + Y1 yalnız havalimanı katmanı (pist katmanı YOK) + LOBT. v222 ile farkı = yalnız pist katmanı.

```
 apt  ay kapsam     n    TF  geçersiz_%  yayımlanan  fk_ort  fk_std  sapma  kayma
EDDF   1  hepsi 15667 15667        12.7       853.5     0.5     2.8   -7.1    0.0
EDDF   7  hepsi 20650 20650         0.0       854.2     0.5     2.8    8.2   -2.5
EDDM   1  hepsi 10953 11033        31.1       755.0    11.4     2.1  -14.9    0.0
EDDM   7  hepsi 15013 15131         0.8       776.3    11.4     2.1   -5.2    0.9
EGLL   1  hepsi 19283 19283         0.0      1334.9     0.9     0.7  -11.6   10.2
EGLL   7  hepsi 20557 20556         0.0      1344.1     0.9     0.7   -5.1    3.8
EHAM   1  hepsi 16232 16321         1.1       982.1    10.8     4.9  -28.5   18.7
EHAM   7  hepsi 21950 22060         0.9       840.9    10.8     4.9   -2.6    0.0
LEBL   1  hepsi 12201 12201         0.0       883.1    -2.4     2.2    3.1   -0.0
LEBL   7  hepsi 17879 17867         0.0       967.2    -2.4     2.2    8.6   -4.2
LEMD   1  hepsi 16802 16787         3.1       987.3    -2.2     4.6   -5.9    0.0
LEMD   7  hepsi 20152 20127         0.0      1020.4    -2.2     4.6   -5.2    0.0
LFPG   1  hepsi 18129 18131         9.8       928.2   -18.9    17.4  -22.9    0.0
LFPG   7  hepsi 21743 21743         0.0       965.1   -18.9    17.4  -33.7    0.0
LIRF   1  hepsi 11061 11061         0.4      1043.3    -9.1     3.9  -10.5    2.7
LIRF   7  hepsi 15838 15834         0.5      1214.0    -9.1     3.9   61.4  -53.5
LSZH   1  hepsi  9995 10012        19.7       711.0     0.6     6.0   -6.9    0.0
LSZH   7  hepsi 13155 13166         0.7       741.5     0.6     6.0    4.0   -0.0
LTFM   1  hepsi 22396 22455         4.8      1088.7   -18.3     8.6   31.1  -14.0
LTFM   7  hepsi 25185 25265         0.0      1008.8   -18.3     8.6   -3.1    0.0
```

- Değişim: RMS 13.00 sn, ort -1.94 sn; LOBT kırpılan 8