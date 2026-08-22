# Audit full benchmark CafeBERT — all seeds

- Rows: **480 / 480 expected**
- Unique benchmark keys: **480**
- Seeds: **11, 29, 42, 47**
- Config SHA-256: `4bbba1f8131d9c8ed741219255d2985be219ecc9b9368ad84e025bcac1cd840b`
- Status: **PASS**

## Coverage by corpus

| Corpus | Rows |
|---|---:|
| vi-medical | 120 |
| vietnamese-news | 120 |
| visfd | 120 |
| vntc-it | 120 |

## Coverage by model

| Model | Rows |
|---|---:|
| bertopic_kmeans | 80 |
| lda | 80 |
| nmf | 80 |
| s3_angular | 80 |
| s3_axial | 80 |
| s3_combined | 80 |

## Coverage by seed

| Seed | Rows |
|---|---:|
| 11 | 120 |
| 29 | 120 |
| 42 | 120 |
| 47 | 120 |

## Document-ID provenance

| Corpus | Documents | SHA-256 of ordered document IDs |
|---|---:|---:|
| vi-medical | 12060 | 15f5601e6cb64d461ded353f61131a7395abd361ca4fce9959f42a19b966d614 |
| vietnamese-news | 858 | 4a69f5df5cc5cee9d68d807eca0f7d5d3c4bd2c42a1637f8e83d3247819a1c7d |
| visfd | 10000 | c719fdf08e863348f772bc3900112d2708ed736eb526037ad3dd5b5e8f2bdfb8 |
| vntc-it | 3571 | 6fbe5bfbdf1b963816a421668db82d426899d6689adcc353fde88d9d6ff61212 |

## Cell coverage

Every corpus × model × seed cell must contain exactly five topic counts (10, 20, 30, 40, 50).

| Corpus | Model | Seed | k values |
|---|---:|---:|---:|
| vi-medical | bertopic_kmeans | 11 | 5 |
| vi-medical | bertopic_kmeans | 29 | 5 |
| vi-medical | bertopic_kmeans | 42 | 5 |
| vi-medical | bertopic_kmeans | 47 | 5 |
| vi-medical | lda | 11 | 5 |
| vi-medical | lda | 29 | 5 |
| vi-medical | lda | 42 | 5 |
| vi-medical | lda | 47 | 5 |
| vi-medical | nmf | 11 | 5 |
| vi-medical | nmf | 29 | 5 |
| vi-medical | nmf | 42 | 5 |
| vi-medical | nmf | 47 | 5 |
| vi-medical | s3_angular | 11 | 5 |
| vi-medical | s3_angular | 29 | 5 |
| vi-medical | s3_angular | 42 | 5 |
| vi-medical | s3_angular | 47 | 5 |
| vi-medical | s3_axial | 11 | 5 |
| vi-medical | s3_axial | 29 | 5 |
| vi-medical | s3_axial | 42 | 5 |
| vi-medical | s3_axial | 47 | 5 |
| vi-medical | s3_combined | 11 | 5 |
| vi-medical | s3_combined | 29 | 5 |
| vi-medical | s3_combined | 42 | 5 |
| vi-medical | s3_combined | 47 | 5 |
| vietnamese-news | bertopic_kmeans | 11 | 5 |
| vietnamese-news | bertopic_kmeans | 29 | 5 |
| vietnamese-news | bertopic_kmeans | 42 | 5 |
| vietnamese-news | bertopic_kmeans | 47 | 5 |
| vietnamese-news | lda | 11 | 5 |
| vietnamese-news | lda | 29 | 5 |
| vietnamese-news | lda | 42 | 5 |
| vietnamese-news | lda | 47 | 5 |
| vietnamese-news | nmf | 11 | 5 |
| vietnamese-news | nmf | 29 | 5 |
| vietnamese-news | nmf | 42 | 5 |
| vietnamese-news | nmf | 47 | 5 |
| vietnamese-news | s3_angular | 11 | 5 |
| vietnamese-news | s3_angular | 29 | 5 |
| vietnamese-news | s3_angular | 42 | 5 |
| vietnamese-news | s3_angular | 47 | 5 |
| vietnamese-news | s3_axial | 11 | 5 |
| vietnamese-news | s3_axial | 29 | 5 |
| vietnamese-news | s3_axial | 42 | 5 |
| vietnamese-news | s3_axial | 47 | 5 |
| vietnamese-news | s3_combined | 11 | 5 |
| vietnamese-news | s3_combined | 29 | 5 |
| vietnamese-news | s3_combined | 42 | 5 |
| vietnamese-news | s3_combined | 47 | 5 |
| visfd | bertopic_kmeans | 11 | 5 |
| visfd | bertopic_kmeans | 29 | 5 |
| visfd | bertopic_kmeans | 42 | 5 |
| visfd | bertopic_kmeans | 47 | 5 |
| visfd | lda | 11 | 5 |
| visfd | lda | 29 | 5 |
| visfd | lda | 42 | 5 |
| visfd | lda | 47 | 5 |
| visfd | nmf | 11 | 5 |
| visfd | nmf | 29 | 5 |
| visfd | nmf | 42 | 5 |
| visfd | nmf | 47 | 5 |
| visfd | s3_angular | 11 | 5 |
| visfd | s3_angular | 29 | 5 |
| visfd | s3_angular | 42 | 5 |
| visfd | s3_angular | 47 | 5 |
| visfd | s3_axial | 11 | 5 |
| visfd | s3_axial | 29 | 5 |
| visfd | s3_axial | 42 | 5 |
| visfd | s3_axial | 47 | 5 |
| visfd | s3_combined | 11 | 5 |
| visfd | s3_combined | 29 | 5 |
| visfd | s3_combined | 42 | 5 |
| visfd | s3_combined | 47 | 5 |
| vntc-it | bertopic_kmeans | 11 | 5 |
| vntc-it | bertopic_kmeans | 29 | 5 |
| vntc-it | bertopic_kmeans | 42 | 5 |
| vntc-it | bertopic_kmeans | 47 | 5 |
| vntc-it | lda | 11 | 5 |
| vntc-it | lda | 29 | 5 |
| vntc-it | lda | 42 | 5 |
| vntc-it | lda | 47 | 5 |
| vntc-it | nmf | 11 | 5 |
| vntc-it | nmf | 29 | 5 |
| vntc-it | nmf | 42 | 5 |
| vntc-it | nmf | 47 | 5 |
| vntc-it | s3_angular | 11 | 5 |
| vntc-it | s3_angular | 29 | 5 |
| vntc-it | s3_angular | 42 | 5 |
| vntc-it | s3_angular | 47 | 5 |
| vntc-it | s3_axial | 11 | 5 |
| vntc-it | s3_axial | 29 | 5 |
| vntc-it | s3_axial | 42 | 5 |
| vntc-it | s3_axial | 47 | 5 |
| vntc-it | s3_combined | 11 | 5 |
| vntc-it | s3_combined | 29 | 5 |
| vntc-it | s3_combined | 42 | 5 |
| vntc-it | s3_combined | 47 | 5 |

## Validation

- All 480 expected configurations are unique, status `ok`, finite and structurally valid.
- Each corpus × model × seed cell contains k = 10, 20, 30, 40, 50.
- Each configuration returned its requested topic count with at least 10 terms per topic.
- Document counts and ordered document-ID hashes are internally consistent for every corpus.
- Timing invariants hold: pipeline time is not below fit-only time, and total cold time is not below pipeline time.
