# Visual QA — CafeBERT benchmark charts

## Checked files

| File | Finding | Status |
|---|---|---|
| `cafebert-wec-in.png` | Four corpus panels are legible; legend maps all six methods; WEC-in values and seed-SD bands show the expected pattern: LDA/NMF are highest on Vietnamese-news, while S³ is above the lexical baselines on UIT-ViSFD, ViMedical and VNTC-CNTT. | Pass |
| `cafebert-diversity.png` | Four corpus panels are legible; method colors match the WEC-in chart; diversity visibly separates the high-diversity S³ variants on UIT-ViSFD/VNTC from the different trade-offs on Vietnamese-news and ViMedical. | Pass |
| `cafebert-fit-timing.png` | Four corpus panels are legible; title and y-axis explicitly label the fit-only stage after representation is ready; all six method abbreviations are visible. The chart must not be interpreted as end-to-end cold-start time. | Pass |

The charts use four-seed means and sample standard-deviation bands. They are descriptive visualizations and do not replace the seed-42 tables or the audited `full_results.csv`.
