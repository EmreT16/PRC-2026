"""adsb.lol globe_history günlük arşivlerini indirir (açık veri, ODbL-1.0).

Kaynak: https://github.com/adsblol/globe_history_<yıl>; her gün için hangi sürümün (prod/staging) kullanılacağı depo kökündeki
PREFERRED_RELEASES.txt'de. Bir gün 2–3 parçalı bölünmüş tar (günlük ~3–4 GB, tüm dünya).
Hedef: PRC_ADSB_KOK/ham/<YYYY-MM-DD>/ (girdi klasörlerine yazılmaz). Biten günde .tamam dosyası.
Kullanım: python scripts/adsb_indir.py 2025-07-15 2026-07-14
          python scripts/adsb_indir.py 2026-07-01:2026-07-31      (aralık)
"""
import datetime as dt
import subprocess
import os
import sys
import urllib.request
from pathlib import Path

HAM = Path(os.environ.get("PRC_ADSB_KOK", str(Path(__file__).resolve().parents[2] / "PRC_adsb"))) / "ham"  # 02.10: D:/PRC_adsb ve C:/PRC_adsb buraya birleştirildi


def gunler(args):
    out = []
    for a in args:
        if ":" in a:
            b, e = (dt.date.fromisoformat(x) for x in a.split(":"))
            out += [b + dt.timedelta(days=i) for i in range((e - b).days + 1)]
        else:
            out.append(dt.date.fromisoformat(a))
    return out


def tercihli(yil):
    f = HAM / f"PREFERRED_RELEASES_{yil}.txt"
    if not f.exists():
        url = f"https://raw.githubusercontent.com/adsblol/globe_history_{yil}/main/PREFERRED_RELEASES.txt"
        f.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _parcalar(d, urls):
    for url in urls:
        hedef = d / url.rsplit("/", 1)[1]
        subprocess.run(["curl", "-L", "--fail", "--retry", "8", "--retry-delay", "10", "-C", "-", "-sS",
                        "-o", str(hedef), url], check=True)


def indir(g, diger=False):
    """diger=True: indirilmiş kopyayı silip öbür kopyayı (prod ↔ staging) indir (arşiv içi bozukluk için)."""
    d = HAM / g.isoformat()
    if (d / ".tamam").exists() and not diger:
        print(f"{g}: zaten var", flush=True)
        return
    etiket = f"/v{g:%Y.%m.%d}-planes-readsb-"
    satir = [ln for ln in tercihli(g.year) if etiket in ln]
    if not satir:  # yıl sonu günü bir sonraki yılın deposunda olabilir (31.12.2025 → globe_history_2026)
        satir = [ln for ln in tercihli(g.year + 1) if etiket in ln]
    if len(satir) != 1:
        raise SystemExit(f"{g}: PREFERRED_RELEASES satırı bulunamadı ({len(satir)})")
    d.mkdir(parents=True, exist_ok=True)
    urls = satir[0].split(",")
    # 09.10 (final ayları veri denetimi): tercih edilen sürümde iki ayrı arşiv karışık olabiliyor
    # (06.06.2026 staging: tek ".tar" 1,88 GB + ".tar.aa / .ab"); birleştirilince yalnız ilki okunuyor (UTC 00–09 eksik)
    # → böyle bir sürümde doğrudan öbür kopya
    karisik = any(u.endswith(".tar") for u in urls) and any(".tar.a" in u for u in urls)
    if karisik and not diger:
        print(f"{g}: tercih edilen sürümde karışık arşiv ({len(urls)} parça) → öbür kopya", flush=True)
        diger = True
    if diger and (d / ".tamam").exists():
        urls = (d / ".tamam").read_text(encoding="utf-8").split(",")
        (d / ".tamam").unlink()
    try:
        if diger:
            raise subprocess.CalledProcessError(1, "diger")
        _parcalar(d, urls)
    except subprocess.CalledProcessError:
        # tercih edilen kopya inmedi (ör. 11.07.2026 staging: yarıda kesilip devam edemedi) → öbür kopyayı dene
        # (bir kopya bozuksa diğeri genelde sağlam). Parça sayısı sürüm sayfasından okunur.
        eski, yeni = ("staging", "prod") if "-staging-" in urls[0] else ("prod", "staging")
        tag = urls[0].rsplit("/", 2)[1].replace(f"-{eski}-", f"-{yeni}-")
        depo = urls[0].split("/")[4]  # globe_history_<yıl>; yıl sonu günlerinde bir sonraki yılın deposu
        api = f"https://api.github.com/repos/adsblol/{depo}/releases/tags/{tag}"
        import json
        varliklar = json.loads(urllib.request.urlopen(api, timeout=60).read())["assets"]
        for p in d.glob("*.tar*"):
            p.unlink()
        urls = [a["browser_download_url"] for a in sorted(varliklar, key=lambda a: a["name"])
                if ".tar" in a["name"]]
        print(f"{g}: {eski} kopyası inmedi, {yeni} deneniyor ({len(urls)} parça)", flush=True)
        _parcalar(d, urls)
    (d / ".tamam").write_text(",".join(urls), encoding="utf-8")
    boyut = sum(p.stat().st_size for p in d.glob("*.tar*")) / 1e9
    print(f"{g}: indirildi, {boyut:.2f} GB", flush=True)


if __name__ == "__main__":
    HAM.mkdir(parents=True, exist_ok=True)
    for g in gunler(sys.argv[1:]):
        indir(g)
