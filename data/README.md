# Data

Daily closing bid/ask quotes provided by SEB for this project: Boliden (BOL SS) and Volvo B (VOLVB SS), the stock and all listed
options, 2 May to 30 Dec 2025. The options are American; the period has no dividend, so SEB's note (docs/seb_data_guide.txt) says
they can be valued with Black-Scholes-type models and a dividend yield of 0.

| File | Content |
|---|---|
| `raw/20260908_085958_BOL SS.csv`, `raw/20260908_090008_VOLVB SS.csv` | SEB's export, semicolon-separated: Date, Name (option series code), Isin, InsType (Option/Stock), CallPut, Expiry, Strike, Bid, Ask |
| `clean/BOL_SS_rensad.xlsx`, `clean/VOLVB_SS_rensad.xlsx` | the same quotes in cleaned form (sheet `Data`); these are what the scripts read |

The files contain market quotes only: instrument codes, prices and dates, no account or personal information.

These data belong to SEB. Keep the repository private unless SEB has agreed to publication.
