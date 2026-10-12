# Gönderim denetimi — strong-moose_v370

- Dosya: `<repo>\gonderim\strong-moose_v370.parquet`
- MD5: `9fe7d12a9bbba1446427bd15ddc7d369`
- Kaynak: `gonderim/strong-moose_v369.parquet` (MD5 bd0a73e9c276cde1066fa0500398e767) + Y1 kalibrasyonu (havalimanı + pist) (scripts/g_kalibre.py) + LOBT penceresi
- Not: strong-moose_v369.parquet + Y1 havalimanı + pist (1σ) + LOBT. strong-moose_v365.parquet + taban TabM 5 tohum (harman D_kqgpPeT2t5; düzeltici v 0.9). Tek fark taban TabM tohum sayısı.

```
 apt  ay                  kapsam     n    TF  geçersiz_%  yayımlanan  fk_ort  fk_std  sapma  kayma
EDDF   1                   hepsi 15667 15667        12.7       853.5     0.5     2.8   -1.1    0.0
EDDF   7                   hepsi 20650 20650         0.0       854.2     0.5     2.8    8.6   -5.8
EDDF   7                pist 07C  3105     0         0.0       784.3    -6.3     5.2   13.2   -0.0
EDDF   7                 pist 18 13092     0         0.0       893.2     0.2     1.0    0.4   -0.0
EDDF   7                pist 25C  4303     0         0.0       768.3     3.0    10.4   19.1   -0.0
EDDM   1                   hepsi 10953 11033        31.1       755.0    11.4     2.1  -11.9    0.0
EDDM   7                   hepsi 15013 15131         0.8       776.3    11.4     2.1   -3.1    1.0
EDDM   7                pist 08L  1450     0         0.0       761.5     5.9     8.8   -2.5    0.0
EDDM   7                pist 08R  2078     0         0.0       885.6    14.7     8.2    0.2   -0.0
EDDM   7                pist 26L  7114     0         0.0       739.1    14.0     8.5   -4.6    0.0
EDDM   7                pist 26R  4371     0         0.0       791.3    12.2     7.9    4.4   -0.0
EGLL   1                   hepsi 19283 19283         0.0      1334.9     0.9     0.7  -11.4   10.7
EGLL   1                pist 09R  8940     0         0.0      1294.2     0.7     1.5    0.4   -0.0
EGLL   1                pist 27L  5266     0         0.0      1348.0     1.1     1.3   -7.1    3.3
EGLL   1                pist 27R  5077     0         0.0      1393.3     1.0     1.4    3.0   -0.0
EGLL   7                   hepsi 20557 20556         0.0      1344.1     0.9     0.7   -0.7    0.1
EGLL   7                pist 09R  7867     0         0.0      1386.7     0.7     1.5    0.6   -0.0
EGLL   7                pist 27L  6406     0         0.0      1294.7     1.1     1.3    0.8   -0.0
EGLL   7                pist 27R  6154     0         0.0      1341.2     1.0     1.4   -5.8    1.7
EHAM   1                   hepsi 16232 16321         1.1       982.1    10.8     4.9  -25.1   20.2
EHAM   1                pist 18L  6074     0         0.0       824.6     8.7     6.3    4.3   -0.0
EHAM   1                 pist 24  5196     0         0.0      1041.5     9.4     5.7  -25.0    7.9
EHAM   1                pist 36L  1895     0         0.0      1354.0     4.4     3.2   -0.7    0.0
EHAM   7                   hepsi 21950 22060         0.9       840.9    10.8     4.9   -0.9    0.0
EHAM   7                pist 18L  1893     0         0.0       703.5     8.7     6.3    2.4   -0.0
EHAM   7                 pist 24  5015     0         0.0       632.4     9.4     5.7    8.7   -0.0
EHAM   7                pist 36C  4911     0         0.0       715.7    -9.2     5.0   -1.2    0.0
EHAM   7                pist 36L  8958     0         0.0      1080.6     4.4     3.2   -1.4    0.0
LEBL   1                   hepsi 12201 12201         0.0       883.1    -2.4     2.2    7.5   -5.3
LEBL   1                pist 06R  2804     0         0.0       822.1    -3.1     2.3    1.0   -0.0
LEBL   1                pist 24L  9135     0         0.0       897.8    -0.3     1.7    1.4   -0.0
LEBL   7                   hepsi 17879 17867         0.0       967.2    -2.4     2.2   18.4  -16.2
LEBL   7                pist 06R  6919     0         0.0       924.6    -3.1     2.3    5.2   -0.0
LEBL   7                pist 24L 10596     0         0.0       989.9    -0.3     1.7   -1.8    0.0
LEMD   1                   hepsi 16802 16787         3.1       987.3    -2.2     4.6   -5.1    0.4
LEMD   1                pist 14L  1776     0         0.0      1049.9     8.5    10.9  -11.0    0.0
LEMD   1                pist 14R  1189     0         0.0       957.0    37.0     9.6   -3.3    0.0
LEMD   1                pist 36L  6691     0         0.0       909.4    -4.0     3.2   -3.1    0.0
LEMD   1                pist 36R  7145     0         0.0      1051.8    -9.1     2.6   -4.3    0.0
LEMD   7                   hepsi 20152 20127         0.0      1020.4    -2.2     4.6   -4.0    0.0
LEMD   7                pist 14L  3722     0         0.0      1089.4     8.5    10.9   24.9   -0.0
LEMD   7                pist 14R  3309     0         0.0      1009.1    37.0     9.6    3.0   -0.0
LEMD   7                pist 36L  7069     0         0.0       949.4    -4.0     3.2  -10.2    0.6
LEMD   7                pist 36R  6049     0         0.0      1073.2    -9.1     2.6   -5.3    0.0
LFPG   1                   hepsi 18129 18131         9.8       928.2   -18.9    17.4  -26.5    9.1
LFPG   1                pist 08L  3951     0         0.0       922.1    -6.9    12.5  -52.9   15.5
LFPG   1                pist 26R  6580     0         0.0       902.9   -14.1    23.8    9.8   -0.0
LFPG   7                   hepsi 21743 21743         0.0       965.1   -18.9    17.4  -36.0   18.6
LFPG   7                pist 08L  6811     0         0.0       956.1    -6.9    12.5  -15.3    0.0
LFPG   7                pist 26R  5506     0         0.0       920.4   -14.1    23.8   -4.8    0.0
LIRF   1                   hepsi 11061 11061         0.4      1043.3    -9.1     3.9   -8.1    4.2
LIRF   1                 pist 25 10166     0         0.0      1026.5    -0.8     1.6   -3.2    0.0
LIRF   7                   hepsi 15838 15834         0.5      1214.0    -9.1     3.9   48.8  -44.9
LIRF   7                pist 16R  1544     0         0.0      1763.8     0.5    14.7  -47.2    3.0
LIRF   7                 pist 25 14036     0         0.0      1176.3    -0.8     1.6  -19.4   14.5
LSZH   1                   hepsi  9995 10012        19.7       711.0     0.6     6.0   -9.3    0.0
LSZH   7                   hepsi 13155 13166         0.7       741.5     0.6     6.0    4.5   -0.0
LSZH   7                 pist 16  1299     0         0.0      1079.3    20.0    15.2   -1.3    0.0
LSZH   7                 pist 28  7738     0         0.0       695.8     6.2     4.2    4.1   -0.0
LSZH   7                 pist 32  3794     0         0.0       738.9    -8.6    13.2   -7.9    0.0
LTFM   1                   hepsi 22396 22455         4.8      1088.7   -18.3     8.6   25.4  -16.8
LTFM   1 pist atlandı (tutarsız)     0     0         0.0         0.0     0.0     0.0    0.0    0.0
LTFM   7                   hepsi 25185 25265         0.0      1008.8   -18.3     8.6   -3.5    0.0
LTFM   7                pist 35L  9880     0         0.0      1103.4    -4.8     5.3   -5.6    0.0
LTFM   7                 pist 36 13018     0         0.0       907.7     5.0    16.3   15.9   -0.0
```

- Değişim: RMS 12.14 sn, ort -0.12 sn; LOBT kırpılan 11