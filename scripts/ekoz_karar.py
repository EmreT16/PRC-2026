"""Tarama sonucuna göre PRC_EKOZ özellik seçimi (02.10 akşam). Girdi: kutuk/tarama_ay.log,
kutuk/tarama_kar.log, kutuk/tarama_kar_yz.log (scripts/tarama_olay.py). Çıktı: kutuk/ekoz_secim.json. Hiçbiri kabul
edilmezse çıkış kodu 1 (zincir durur).
Kurallar (gürültü tabanı: ay kipinde ayda ±0,1; kar kipinde kar satırlarında katta ±1,4, toplamda ±0,2):
  YZ / NMG : ay kipinde ort. ≤ −0,10 ve ≥ 5/6 ay iyi; kar kipinde karsız ≤ +0,02 (varsa).
  KAR grupları : kar kipinde kar ≤ −3,0 ve karsız ≤ +0,02; ay kipinde KARHEP ort. ≤ +0,05. KARHEP, KAR ve KARG'den en iyisi.
  SIC : kar kriteri + ay kipinde ort. ≤ −0,10 ve ≥ 5/6 (sıcaklık ay vekili olabilir).
"""
import json
import re
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1] / "kutuk"
GRUP = {"YZ": ("yuzey_oz.parquet", ["ARR_TI_W30_MED", "ARR_TI_W30_MAX", "ARR_TI_W30_N", "ARR_TI_W15_MED", "ARR_TI_F15_MED"]),
        "NMG": ("yuzey_oz.parquet", ["NMT_C15_G", "NMT_F15_G", "NMT_PIST_C15_G", "NMT_YAPAY"]),
        "KAR": ("kar_oz.parquet", ["KAR_3S", "KAR_6S", "KAR_12S", "KAR_24S", "DON_6S", "KAR_SONRA_3S"]),
        "KARG": ("kar_oz.parquet", ["KAR_GUN", "KAR_ONCEKI_GUN"]),
        "SIC": ("kar_oz.parquet", ["SIC_0", "SIC_MIN12", "SPREAD_0"])}
GRUP["KARHEP"] = ("kar_oz.parquet", GRUP["KAR"][1] + GRUP["KARG"][1])


def oku():
    ay, kar = {}, {}
    for line in (K / "tarama_ay.log").read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\w+): .*\| ort ([+-][\d.]+) \| iyi (\d)/6", line)
        if m:
            ay[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    for f in ("tarama_kar.log", "tarama_kar_yz.log"):
        if (K / f).exists():
            for line in (K / f).read_text(encoding="utf-8").splitlines():
                m = re.match(r"(\w+): kar ([+-][\d.]+) .*\| karsız ([+-][\d.]+)", line)
                if m:
                    kar[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return ay, kar


def main():
    ay, kar = oku()
    print("ay:", ay, "\nkar:", kar)
    sec = []
    for g in ("YZ", "NMG"):
        if g in ay and ay[g][0] <= -0.10 and ay[g][1] >= 5 and kar.get(g, (0, 0))[1] <= 0.02:
            sec.append(g)
    ay_kar_ok = ay.get("KARHEP", (9, 0))[0] <= 0.05
    aday = [(kar[g][0], g) for g in ("KARHEP", "KAR", "KARG") if g in kar and kar[g][0] <= -3.0 and kar[g][1] <= 0.02]
    if aday and ay_kar_ok:
        sec.append(min(aday)[1])
    if ("SIC" in kar and kar["SIC"][0] <= -3.0 and kar["SIC"][1] <= 0.02 and ay.get("SIC", (9, 0))[0] <= -0.10
            and ay.get("SIC", (0, 0))[1] >= 5):
        sec.append("SIC")
    out = {}
    for g in sec:
        f, cols = GRUP[g]
        out.setdefault(f, [])
        out[f] += [c for c in cols if c not in out[f]]
    print("seçilen gruplar:", sec, "→", out)
    if not out:
        print("KABUL YOK")
        sys.exit(1)
    (K / "ekoz_secim.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
