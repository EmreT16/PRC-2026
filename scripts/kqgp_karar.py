"""01.10: pist bazlı gecikme endeksi kararı — lgbh3_kqgp ile lgbh3_kqg'nin bölme kütüklerindeki aylık taksi RMSE'leri.
Kural: 6 ayın en az 5'inde iyi ve ortalama Δ ≤ −0,15 sn → KABUL (çıkış 0), değilse RET (çıkış 1)."""
import re
import sys


def aylar(yol):
    d = {}
    for blok in re.split(r"\n== ", open(yol, encoding="utf-8").read()):
        for m in re.finditer(r"ay +(\d+): taban [\d.]+ \| aday ([\d.]+)", blok):
            d.setdefault(int(m.group(1)), float(m.group(2)))
    return d


sys.stdout.reconfigure(encoding="utf-8")
a, b = aylar("kutuk/kqg_lgb_split123.log"), aylar("kutuk/kqgp_lgb_split123.log")
assert sorted(a) == sorted(b) == [1, 2, 6, 7, 8, 12], (sorted(a), sorted(b))
fark = {m: b[m] - a[m] for m in sorted(a)}
iyi, ort = sum(v < 0 for v in fark.values()), sum(fark.values()) / 6
print("lgbh3 kqg → kqgp (taksi RMSE, sn):", " ".join(f"{m}: {a[m]:.2f} → {b[m]:.2f} ({fark[m]:+.2f})" for m in sorted(a)))
kabul = iyi >= 5 and ort <= -0.15
print(f"iyi ay {iyi}/6 | ortalama Δ {ort:+.3f} → {'KABUL' if kabul else 'RET'}")
sys.exit(0 if kabul else 1)
