# CafeBERT checkpoint and model artifacts

The full benchmark uses the public pretrained checkpoint [`uitnlp/CafeBERT`](https://huggingface.co/uitnlp/CafeBERT) at revision `af76fcf2a04096b2b54b348a3e4eb48253c93c5d`. This revision was the `main` revision verified on 2026-08-22. Run the command below to materialize the exact snapshot locally and write a per-file SHA-256 manifest:

```bash
python -m benchmark.cafebert_full.fetch_cafebert_checkpoint \
  --output-dir benchmark/cafebert_full/pretrained/CafeBERT
```

Then run the benchmark against that local snapshot:

```bash
export S3_CAFEBERT_CHECKPOINT_DIR="$PWD/benchmark/cafebert_full/pretrained/CafeBERT"
```

`checkpoint_manifest.json` records every local file, byte size and SHA-256. The model weights are intentionally ignored by Git and are not redistributed in this repository.

There is **no S³ pretrained checkpoint** in this experiment. S³ is fit per corpus after CafeBERT embeddings are created. LDA and NMF use CountVectorizer; BERTopic uses CafeBERT embeddings with UMAP plus KMeans. Do not label the downloaded CafeBERT snapshot as an S³ checkpoint.
