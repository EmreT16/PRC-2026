# Gönderim denetimi — strong-moose_v226

- Dosya: `<repo>\gonderim\strong-moose_v226.parquet`
- MD5: `6cdb2280c8747173fe998243b5fb11cf`
- Kaynak: `gonderim/strong-moose_v223.parquet` (MD5 615fe6b9b81d2b109e31caa584f55d46) + Y1 kalibrasyonu (scripts/g_kalibre.py) + LOBT penceresi
- Not: strong-moose_v223.parquet + Y1 yalnız havalimanı katmanı (pist katmanı YOK) + LOBT. v224 ile farkı = yalnız pist katmanı.

```
 apt  ay kapsam     n    TF  geçersiz_%  yayımlanan  fk_ort  fk_std  sapma  kayma
EDDF   1  hepsi 15667 15667        12.7       853.5     0.5     2.8   -8.1    0.0
EDDF   7  hepsi 20650 20650         0.0       854.2     0.5     2.8    8.6   -2.9
EDDM   1  hepsi 10953 11033        31.1       755.0    11.4     2.1  -15.0    0.0
EDDM   7  hepsi 15013 15131         0.8       776.3    11.4     2.1   -5.7    1.5
EGLL   1  hepsi 19283 19283         0.0      1334.9     0.9     0.7  -11.8   10.4
EGLL   7  hepsi 20557 20556         0.0      1344.1     0.9     0.7   -5.1    3.7
EHAM   1  hepsi 16232 16321         1.1       982.1    10.8     4.9  -28.2   18.4
EHAM   7  hepsi 21950 22060         0.9       840.9    10.8     4.9   -2.3    0.0
LEBL   1  hepsi 12201 12201         0.0       883.1    -2.4     2.2    3.3   -0.0
LEBL   7  hepsi 17879 17867         0.0       967.2    -2.4     2.2    8.6   -4.1
LEMD   1  hepsi 16802 16787         3.1       987.3    -2.2     4.6   -6.0    0.0
LEMD   7  hepsi 20152 20127         0.0      1020.4    -2.2     4.6   -4.7    0.0
LFPG   1  hepsi 18129 18131         9.8       928.2   -18.9    17.4  -22.6    0.0
LFPG   7  hepsi 21743 21743         0.0       965.1   -18.9    17.4  -33.9    0.0
LIRF   1  hepsi 11061 11061         0.4      1043.3    -9.1     3.9  -10.5    2.6
LIRF   7  hepsi 15838 15834         0.5      1214.0    -9.1     3.9   61.3  -53.4
LSZH   1  hepsi  9995 10012        19.7       711.0     0.6     6.0   -7.1    0.0
LSZH   7  hepsi 13155 13166         0.7       741.5     0.6     6.0    3.8   -0.0
LTFM   1  hepsi 22396 22455         4.8      1088.7   -18.3     8.6   31.6  -14.4
LTFM   7  hepsi 25185 25265         0.0      1008.8   -18.3     8.6   -5.3    0.0
```

- Değişim: RMS 13.00 sn, ort -1.96 sn; LOBT kırpılan 8