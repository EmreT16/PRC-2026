# A3 — v126 özellik hattının parite testi

Eğitim Ocak+Temmuz 2025 (339,011 satır) ↔ test Ocak+Temmuz 2026 (339,551 satır); 134 özellik. Eşikler: `src/parity.py`.

## Ham değerler: {'tamam': 130, 'UYARI': 4}

| Özellik | Seviye | Neden |
|---|---|---|
| `CALLSIGN_flt` | UYARI | test satırlarının 46.9%'i eğitimde görülmemiş değer |
| `FLIGHT_CODE` | UYARI | test satırlarının 39.3%'i eğitimde görülmemiş değer |
| `NM_MISSING` | UYARI | iki tarafta da sabit (ölü) |
| `RWY_DISPLACED_THRESH_FT` | UYARI | ortalama farkı 0.55 std; KS 0.20 |

## v126 model matrisi: {'tamam': 126, 'UYARI': 8}

| Özellik | Seviye | Neden |
|---|---|---|
| `ADES_mvt` | UYARI | iki tarafta da sabit (ölü) |
| `CALLSIGN_flt` | UYARI | iki tarafta da sabit (ölü) |
| `MARKET_SEGMENT_flt` | UYARI | iki tarafta da sabit (ölü) |
| `FLIGHT_CODE` | UYARI | testte 2 sonsuz değer |
| `AIRLINE_LETTERS` | UYARI | iki tarafta da sabit (ölü) |
| `ADES_FILED` | UYARI | iki tarafta da sabit (ölü) |
| `NM_MISSING` | UYARI | iki tarafta da sabit (ölü) |
| `ROUTE_PAIR` | UYARI | iki tarafta da sabit (ölü) |
