"""
10 kalkis havalimaninin sabit bilgileri: saat dilimi, ulke kodu (tatil icin),
pist yon acilari (kalkis/inis rotalarindan turetilen aci) ve gece kisit saatleri.

Bu dosya elle hazirlanmis bir referans tablosudur
ve 2.9'daki kararlarin kod karsiligidir.
"""

# ICAO kodu -> (saat dilimi, ulke kodu [holidays kutuphanesi icin])
AIRPORT_INFO = {
    'EDDF': ('Europe/Berlin', 'DE'),
    'EDDM': ('Europe/Berlin', 'DE'),
    'EGLL': ('Europe/London', 'GB'),
    'EHAM': ('Europe/Amsterdam', 'NL'),
    'LEBL': ('Europe/Madrid', 'ES'),
    'LEMD': ('Europe/Madrid', 'ES'),
    'LFPG': ('Europe/Paris', 'FR'),
    'LIRF': ('Europe/Rome', 'IT'),
    'LSZH': ('Europe/Zurich', 'CH'),
    'LTFM': ('Europe/Istanbul', 'TR'),
}

# ICAO bolge oneki (ilk 1-2 harf) -> ulke kodu. Varis havalimanindaki (ADES_mvt)
# tatil bilgisini bulmak icin kullanilir - 911 farkli havalimani icin elle esleme
# yapilamayacagindan oneke gore otomatik tespit yapiliyor.
# En sik gorulen bolgeler once, kapsam genisletilebilir.
ICAO_PREFIX_TO_COUNTRY = {
    'ED': 'DE', 'ET': 'DE',                     # Almanya (sivil + askeri onek)
    'LE': 'ES', 'GC': 'ES',                     # Ispanya (+ Kanarya Adalari)
    'LI': 'IT',                                 # Italya
    'EG': 'GB',                                 # Ingiltere
    'LF': 'FR', 'LFA': 'FR',                    # Fransa
    'LT': 'TR',                                 # Turkiye
    'LS': 'CH',                                 # Isvicre
    'LP': 'PT',                                 # Portekiz
    'EH': 'NL',                                 # Hollanda
    'EP': 'PL',                                 # Polonya
    'EI': 'IE',                                 # Irlanda
    'EK': 'DK',                                 # Danimarka
    'ES': 'SE',                                 # Isvec
    'EN': 'NO',                                 # Norvec
    'EF': 'FI',                                 # Finlandiya
    'LO': 'AT',                                 # Avusturya
    'EB': 'BE',                                 # Belcika
    'LK': 'CZ',                                 # Cekya
    'LH': 'HU',                                 # Macaristan
    'LR': 'RO',                                 # Romanya
    'LB': 'BG',                                 # Bulgaristan
    'LG': 'GR',                                 # Yunanistan
    'LC': 'CY',                                 # Kibris
    'LM': 'MT',                                 # Malta
    'LD': 'HR',                                 # Hirvatistan
    'LJ': 'SI',                                 # Slovenya
    'LZ': 'SK',                                 # Slovakya
    'EV': 'LV',                                 # Letonya
    'EY': 'LT',                                 # Litvanya
    'EE': 'EE',                                 # Estonya
    'UU': 'RU', 'UL': 'RU', 'US': 'RU',         # Rusya (kaba)
    'UK': 'UA',                                 # Ukrayna
    'LY': 'RS',                                 # Sirbistan
    'OM': 'AE',                                 # BAE
    'OT': 'QA',                                 # Katar
    'OE': 'SA',                                 # Suudi Arabistan
    'OJ': 'JO',                                 # Urdun
    'LL': 'IL',                                 # Israil
    'HE': 'EG',                                 # Misir
    'GM': 'MA',                                 # Fas
    'DT': 'TN',                                 # Tunus
    'FA': 'ZA',                                 # Guney Afrika
    'K': 'US',                                  # ABD (tek harf onek, ozel isleniyor)
    'C': 'CA',                                  # Kanada (tek harf onek)
    'MM': 'MX',                                 # Meksika
    'SB': 'BR', 'SD': 'BR', 'SN': 'BR',         # Brezilya
    'SA': 'AR',                                 # Arjantin
    'RJ': 'JP', 'RO': 'JP',                     # Japonya
    'RK': 'KR',                                 # Guney Kore
    'ZB': 'CN', 'ZG': 'CN', 'ZS': 'CN', 'ZU': 'CN', 'ZP': 'CN', 'ZL': 'CN', 'ZJ': 'CN', 'ZH': 'CN', 'ZW': 'CN', 'ZY': 'CN',
    'VH': 'HK',                                 # Hong Kong
    'VT': 'TH',                                 # Tayland
    'WS': 'SG',                                 # Singapur
    'WM': 'MY',                                 # Malezya
    'WI': 'ID',                                 # Endonezya
    'RP': 'PH',                                 # Filipinler
    'VA': 'IN', 'VE': 'IN', 'VI': 'IN', 'VO': 'IN',  # Hindistan
    'OP': 'PK',                                 # Pakistan
    'YB': 'AU', 'YM': 'AU', 'YP': 'AU', 'YS': 'AU', 'YW': 'AU',  # Avustralya
    'NZ': 'NZ',                                 # Yeni Zelanda
}


def get_country_from_icao(icao_series):
    """ADES_mvt gibi bir ICAO kodu serisinden ulke kodu cikarir.

    Once 2 harfli onege bakar, bulamazsa 1 harfli onege (ABD/Kanada icin) duser.
    Eslesme yoksa None/NaN birakir - holidays kutuphanesi zaten o satirlar icin
    NaN uretecek, zorlamaya gerek yok (kabul edilen risk).
    """
    import pandas as pd
    s = icao_series.astype(str)
    two = s.str[:2].map(ICAO_PREFIX_TO_COUNTRY)
    one = s.str[:1].map(ICAO_PREFIX_TO_COUNTRY)
    return two.fillna(one)


# Bilinen gece ucus kisit saatleri (yerel saat, yaklasik).
# (baslangic_saat, bitis_saat) - gece yarisini gecenler icin baslangic > bitis.
NIGHT_CURFEW = {
    'EDDF': (23, 5),
    'LSZH': (23, 6),   # 23:30 pratikte, saat cozunurlugunde 23 alinir
    'EGLL': (23, 6),
    'EHAM': (23, 6),
    'EDDM': (22, 6),
}
