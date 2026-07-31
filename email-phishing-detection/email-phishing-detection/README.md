# Email Phishing Detection Using Machine Learning


## Project structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── README.md              # how to download the dataset (not committed)
├── src/
│   ├── preprocessing.py       # load, clean, TF-IDF feature extraction
│   └── train.py               # train + compare all three models, save figures
├── notebooks/
│   └── phishing_detection.ipynb   # interactive walkthrough
├── results/
│   ├── metrics.csv
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── confusion_matrices.png
└── report/
    └── Email_Phishing_Detection_IEEE_Report.pdf
```

## Setup

```bash
git clone https://github.com/<your-username>/email-phishing-detection.git
cd email-phishing-detection
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

Then download the dataset into `data/` (see [data/README.md](data/README.md)).

## Usage

Train all three models and regenerate the metrics table and figures:

```bash
python src/train.py --data data/CEAS_08.csv --outdir results
```

Optional flags: `--test-size`, `--random-state`, `--max-features`.

Or open the notebook for a step-by-step version:

```bash
jupyter notebook notebooks/phishing_detection.ipynb
```

## Method

1. **Preprocessing** (`src/preprocessing.py`) — combine `subject` + `body`
   into one text field, filling missing values *before* concatenation, then
   drop empty and duplicate emails.
2. **Features** — TF-IDF with English stop-word removal and a 5,000-term
   vocabulary, kept as a sparse matrix.
3. **Models** — Multinomial Naive Bayes, Logistic Regression, and a linear
   SVM, trained under an identical stratified 80/20 split (`random_state=42`).
4. **Evaluation** — accuracy, precision, recall, F1, and ROC-AUC, plus ROC
   curves and confusion matrices.

## Limitations

- The legitimate emails in CEAS_08 are dominated by technical mailing-list
  traffic, so the near-perfect scores partly reflect a "mailing-list vs.
  everything else" split and may not transfer to a general inbox.
- The number of URLs per email is **not** a useful signal in this corpus
  (nearly identical across classes); message length is far more informative.

## License

Released under the [MIT License](LICENSE).
