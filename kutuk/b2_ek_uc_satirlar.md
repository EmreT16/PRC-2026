# B2 eki — Uç satırların yapısı

Hata = B1 harman ofset tahmini (60 sn alt sınırlı) − gerçek ofset. "Kilitli": \|BLOCK − plan zamanı\| ≤ 60 sn. MSE payı ay ağırlıklı.

## split_1 (339,046 satır, ağırlıklı RMSE 226.951)

### 1. BLOCK kilidi

| Dilim | Satır | Satır payı | MSE payı | RMSE | Ort. hata |
|---|---:|---:|---:|---:|---:|
| BLOCK ≈ SCHED | 30,169 | 8.90% | 14.5% | 289 | +68 |
| BLOCK ≈ EOBT1 | 40,665 | 11.99% | 8.3% | 189 | +57 |
| BLOCK ≈ LOBT | 36,022 | 10.62% | 10.1% | 220 | +55 |
| BLOCK ≈ IOBT | 36,131 | 10.66% | 10.1% | 221 | +55 |
| BLOCK ≈ AOBT_3 (\|y\| ≤ 60) | 69,656 | 20.54% | 7.0% | 132 | -16 |

### 2. Kilitli ve etkili satırlar (BLOCK ≈ SCHED, \|SCHED − AOBT_3\| > 600)

| Dilim | Satır | Satır payı | MSE payı | RMSE | Ort. hata |
|---|---:|---:|---:|---:|---:|
| \|SCHED − AOBT_3\| > 600 | 165,858 | 48.92% | 75.1% | 281 | -2 |
| … bunların kilitli olanları | 2,752 | 0.81% | 11.6% | 854 | +465 |
| … kilitli olmayanları | 163,106 | 48.11% | 63.6% | 260 | -10 |

- Taban oranı: \|SCHED − AOBT_3\| > 600 olan satırların %1.7'i kilitli.
- **Kâhin üst sınırı:** kilitli satırlar kusursuz bilinip tahmin SCHED − AOBT_3 yapılsaydı ağırlıklı RMSE 226.951 → 213.452 (Δ -13.499). Gerçekleşemez; sınıflandırıcı bunun bir kısmını alabilir.

| Havalimanı | Satır | \|SCHED − AOBT_3\| > 600 | Kilitli | Taban oranı | Kilitlilerin MSE payı |
|---|---:|---:|---:|---:|---:|
| EDDF | 36,231 | 17,549 | 24 | 0.1% | 0.1% |
| EDDM | 26,854 | 12,328 | 189 | 1.5% | 0.1% |
| EGLL | 39,744 | 16,657 | 142 | 0.9% | 0.3% |
| EHAM | 40,141 | 19,754 | 18 | 0.1% | 0.0% |
| LEBL | 28,490 | 13,756 | 348 | 2.5% | 1.5% |
| LEMD | 35,019 | 16,720 | 70 | 0.4% | 0.1% |
| LFPG | 38,732 | 22,371 | 182 | 0.8% | 0.3% |
| LIRF | 26,131 | 13,858 | 1,152 | 8.3% | 8.5% |
| LSZH | 21,822 | 12,334 | 28 | 0.2% | 0.1% |
| LTFM | 45,882 | 20,531 | 599 | 2.9% | 0.5% |

### 3. En kötü %5'lik hata diliminin bileşimi

| Özellik | Dilim içindeki satır payı | Tüm satırlarda |
|---|---:|---:|
| BLOCK ≈ SCHED | 9.3% | 8.9% |
| Kilitli ve etkili | 7.0% | 0.8% |
| \|SCHED − AOBT_3\| > 600 | 77.7% | 48.9% |
| Gerçek ofset < 0 | 65.3% | 43.2% |
| \|gerçek ofset\| > 600 | 52.4% | 7.6% |
| AOBT_3 tam dakika değil | 1.7% | 2.4% |

### 4. AOBT_3 tam dakika bayrağı (havalimanı bazında)

| Havalimanı | n (tam değil) | Ort. ofset tam değil / tam | Ort. hata tam değil / tam | RMSE tam değil / tam |
|---|---:|---:|---:|---:|
| EDDF | 398 | +63 / +52 | +9 / -4 | 154 / 180 |
| EDDM | 11 | +467 / -96 | -51 / +5 | 395 / 167 |
| EGLL | 6 | +497 / +13 | -114 / -13 | 242 / 236 |
| EHAM | 7,080 | +170 / +155 | -3 / -0 | 160 / 168 |
| LEBL | 88 | -25 / -71 | -30 / +3 | 261 / 222 |
| LEMD | 35 | -69 / -30 | +50 / -8 | 368 / 172 |
| LFPG | 93 | -85 / -54 | -8 / -11 | 431 / 246 |
| LIRF | 42 | -310 / -249 | -10 / -6 | 385 / 383 |
| LSZH | 3 | -82 / -64 | +73 / -1 | 93 / 183 |
| LTFM | 328 | +2395 / +201 | +32 / -2 | 349 / 246 |

- **F1 üst sınırı:** tam dakika olmayan satırlar, aynı havalimanı ve aydaki tam dakika satırları kadar iyi tahmin edilseydi ağırlıklı RMSE 226.951 → 225.832 (Δ -1.119).

### 5. En büyük 15 hata (sn)

| Havalimanı | Ay | Gerçek ofset | Tahmin (alt sınırlı) | MVT − AOBT_3 | BLOCK − SCHED | SCHED − AOBT_3 | BLOCK − EOBT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSZH | 1 | -8539 | -2163 | 546 | +8021 | -16560 | +1121 |
| LFPG | 7 | -14401 | -8046 | 658 | +1799 | -16200 | -8401 |
| LFPG | 1 | -7622 | -1346 | 941 | +1738 | -9360 | -3602 |
| LTFM | 7 | -5704 | -354 | 1204 | +1676 | -7380 | -2704 |
| LIRF | 7 | -5758 | -411 | 898 | +2 | -5760 | -5218 |
| EHAM | 7 | -3176 | +2124 | 7796 | +1804 | -4980 | -2096 |
| LIRF | 7 | -5640 | -398 | 1320 | +0 | -5640 | -2400 |
| EHAM | 7 | -5996 | -783 | 715 | +64 | -6060 | -1736 |
| LTFM | 7 | -5156 | -452 | 1135 | +1564 | -6720 | -2336 |
| LSZH | 7 | -4941 | -365 | 836 | +519 | -5460 | -3441 |
| LSZH | 1 | -6239 | -1688 | 3299 | +481 | -6720 | +481 |
| EGLL | 7 | +6476 | +2026 | 8163 | +6476 | +0 | +2094 |
| LIRF | 7 | -5105 | -675 | 845 | -5 | -5100 | -4505 |
| LIRF | 7 | -4977 | -611 | 896 | +3 | -4980 | -4197 |
| LIRF | 7 | -10443 | -6245 | 722 | -3 | -10440 | -7983 |

## split_2 (331,474 satır, ağırlıklı RMSE 219.184)

### 1. BLOCK kilidi

| Dilim | Satır | Satır payı | MSE payı | RMSE | Ort. hata |
|---|---:|---:|---:|---:|---:|
| BLOCK ≈ SCHED | 32,621 | 9.84% | 15.4% | 275 | +64 |
| BLOCK ≈ EOBT1 | 42,312 | 12.76% | 8.3% | 177 | +55 |
| BLOCK ≈ LOBT | 37,866 | 11.42% | 9.9% | 205 | +50 |
| BLOCK ≈ IOBT | 37,987 | 11.46% | 10.0% | 205 | +50 |
| BLOCK ≈ AOBT_3 (\|y\| ≤ 60) | 70,094 | 21.15% | 6.9% | 125 | -10 |

### 2. Kilitli ve etkili satırlar (BLOCK ≈ SCHED, \|SCHED − AOBT_3\| > 600)

| Dilim | Satır | Satır payı | MSE payı | RMSE | Ort. hata |
|---|---:|---:|---:|---:|---:|
| \|SCHED − AOBT_3\| > 600 | 147,567 | 44.52% | 72.1% | 279 | +6 |
| … bunların kilitli olanları | 2,774 | 0.84% | 11.7% | 822 | +431 |
| … kilitli olmayanları | 144,793 | 43.68% | 60.4% | 258 | -2 |

- Taban oranı: \|SCHED − AOBT_3\| > 600 olan satırların %1.9'i kilitli.
- **Kâhin üst sınırı:** kilitli satırlar kusursuz bilinip tahmin SCHED − AOBT_3 yapılsaydı ağırlıklı RMSE 219.184 → 205.986 (Δ -13.198). Gerçekleşemez; sınıflandırıcı bunun bir kısmını alabilir.

| Havalimanı | Satır | \|SCHED − AOBT_3\| > 600 | Kilitli | Taban oranı | Kilitlilerin MSE payı |
|---|---:|---:|---:|---:|---:|
| EDDF | 36,300 | 16,342 | 27 | 0.2% | 0.1% |
| EDDM | 25,780 | 10,518 | 137 | 1.3% | 0.1% |
| EGLL | 38,634 | 13,119 | 133 | 1.0% | 0.1% |
| EHAM | 39,855 | 17,452 | 20 | 0.1% | 0.0% |
| LEBL | 28,494 | 12,255 | 273 | 2.2% | 1.2% |
| LEMD | 33,787 | 14,731 | 84 | 0.6% | 0.2% |
| LFPG | 38,361 | 19,884 | 197 | 1.0% | 0.3% |
| LIRF | 25,907 | 12,534 | 1,173 | 9.4% | 8.8% |
| LSZH | 21,271 | 11,083 | 21 | 0.2% | 0.0% |
| LTFM | 43,085 | 19,649 | 709 | 3.6% | 0.9% |

### 3. En kötü %5'lik hata diliminin bileşimi

| Özellik | Dilim içindeki satır payı | Tüm satırlarda |
|---|---:|---:|
| BLOCK ≈ SCHED | 10.8% | 9.8% |
| Kilitli ve etkili | 7.5% | 0.8% |
| \|SCHED − AOBT_3\| > 600 | 72.5% | 44.5% |
| Gerçek ofset < 0 | 64.7% | 43.0% |
| \|gerçek ofset\| > 600 | 49.8% | 6.9% |
| AOBT_3 tam dakika değil | 1.4% | 1.9% |

### 4. AOBT_3 tam dakika bayrağı (havalimanı bazında)

| Havalimanı | n (tam değil) | Ort. ofset tam değil / tam | Ort. hata tam değil / tam | RMSE tam değil / tam |
|---|---:|---:|---:|---:|
| EDDF | 317 | +82 / +68 | +31 / -0 | 174 / 153 |
| EDDM | 14 | +520 / -73 | -206 / -2 | 702 / 154 |
| EGLL | 2 | -26 / +19 | +522 / +3 | 708 / 203 |
| EHAM | 5,151 | +188 / +150 | -2 / -4 | 145 / 149 |
| LEBL | 42 | +56 / -59 | -33 / -0 | 215 / 204 |
| LEMD | 25 | +13 / -31 | -9 / +7 | 273 / 164 |
| LFPG | 87 | -107 / -34 | -41 / +6 | 258 / 216 |
| LIRF | 106 | -189 / -207 | +20 / -5 | 508 / 350 |
| LSZH | 2 | -510 / -59 | +299 / -0 | 312 / 156 |
| LTFM | 433 | +1464 / +197 | +59 / +9 | 544 / 315 |

- **F1 üst sınırı:** tam dakika olmayan satırlar, aynı havalimanı ve aydaki tam dakika satırları kadar iyi tahmin edilseydi ağırlıklı RMSE 219.184 → 217.675 (Δ -1.508).

### 5. En büyük 15 hata (sn)

| Havalimanı | Ay | Gerçek ofset | Tahmin (alt sınırlı) | MVT − AOBT_3 | BLOCK − SCHED | SCHED − AOBT_3 | BLOCK − EOBT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EGLL | 2 | -5765 | +814 | 3485 | +2695 | -8460 | -3485 |
| LIRF | 8 | +6484 | +1002 | 8035 | +664 | +5820 | -116 |
| LFPG | 8 | -5946 | -1148 | 1289 | +294 | -6240 | -1206 |
| LIRF | 8 | -5165 | -444 | 1025 | -5 | -5160 | -3605 |
| LIRF | 8 | -10257 | -5544 | 1076 | -357 | -9900 | -357 |
| LIRF | 8 | -5582 | -1201 | 722 | -2 | -5580 | -3362 |
| LTFM | 2 | -3477 | +870 | 2936 | +1983 | -5460 | -2517 |
| LIRF | 8 | -5160 | -857 | 1019 | +0 | -5160 | -1140 |
| LTFM | 2 | -3177 | +1122 | 2997 | -237 | -2940 | -3537 |
| LTFM | 2 | -4378 | -91 | 1497 | +962 | -5340 | -238 |
| LIRF | 8 | -4495 | -465 | 895 | +5 | -4500 | -3295 |
| LIRF | 8 | -4383 | -408 | 843 | -3 | -4380 | -4143 |
| LTFM | 2 | -4197 | -227 | 2637 | +5883 | -10080 | -1317 |
| LIRF | 8 | -4923 | -979 | 1203 | -3 | -4920 | -2283 |
| LTFM | 8 | +8614 | +4682 | 10241 | +356 | +8258 | +356 |
