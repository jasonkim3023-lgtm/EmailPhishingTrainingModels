# Data

The dataset is **not committed** to this repository (it is ~60 MB and is
redistributed under Kaggle's terms). Download it yourself:

1. Go to the
   [Phishing Email Dataset on Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset).
2. Download and unzip it.
3. Place the `CEAS_08.csv` file in this folder, so the path is:

   ```
   data/CEAS_08.csv
   ```

The training script defaults to that location:

```bash
python src/train.py --data data/CEAS_08.csv --outdir results
```

## Expected columns

`CEAS_08.csv` contains: `sender`, `receiver`, `date`, `subject`, `body`,
`urls`, and `label` (1 = phishing, 0 = legitimate). Only `subject`, `body`,
and `label` are used by the current pipeline.
