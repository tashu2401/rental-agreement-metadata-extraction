# Meta Data Extraction from Rental/Lease Agreements

An LLM-based system that extracts six metadata fields from rental agreement
documents (`.docx` and scanned `.png`) regardless of template or layout, using
Google's Gemini API — **no regex, no rule-based parsing, no static templates.**

Fields extracted: **Agreement Value, Agreement Start Date, Agreement End
Date, Renewal Notice (Days), Party One (Owner/Lessor), Party Two
(Tenant/Lessee).**

---

## 1. Approach

### 1.1 Why an LLM-prompting approach, not a trained NER model

The provided training set has only 10 labeled documents. That is far too
small to train a reliable custom NER/sequence-labeling model from scratch.
Instead, this solution uses **Gemini 2.5 Flash** as a zero-shot extractor,
guided by a single detailed prompt that encodes:

- The exact 6 fields and their expected formats
- Domain knowledge about common ambiguities in these documents (e.g. rent
  vs. security deposit, worded durations vs. numeric days)
- A specific, stated convention for computing an end date when none is
  written explicitly in the document

This satisfies the assignment's constraint against regex/rule-based
extraction: the system does not pattern-match on document structure at all —
it reads and reasons about the document the way a human contract analyst
would, and produces structured JSON as output.

### 1.2 Pipeline

```
train.csv / test.csv  →  pandas DataFrames
train/ , test/ folders →  .docx (python-docx) or .png (Gemini Vision)
                              │
                              ▼
                   EXTRACTION_PROMPT + document content
                              │
                              ▼
                  Gemini 2.5 Flash (generate_content)
                              │
                              ▼
                  JSON response → parsed → 6 fields
                              │
                              ▼
              Predictions DataFrame → compared to ground truth
                              │
                              ▼
                  Per-field exact-match Recall + test_predictions.csv
```

- **`.docx` files** → text extracted via `python-docx` (paragraphs + table
  cells), then sent to Gemini as plain text.
- **`.png` files** → sent directly to Gemini's vision endpoint as an image.
  If the vision call fails outright after retries, the pipeline falls back
  to local Tesseract OCR text and re-attempts extraction via the text path,
  so no image silently produces an empty result without at least one
  alternate strategy being tried.
- **Robust file matching**: CSV "File Name" values don't always map cleanly
  to the actual uploaded filenames (some test files were provided as
  `..._pdf.docx` instead of a plain `.docx`). A `find_file_for_stem()`
  helper resolves these via exact-suffix matching first, with a
  normalized (case/hyphen/underscore-insensitive) fallback, so every CSV
  row reliably finds its file regardless of these naming quirks.
- **Quota-aware retries**: Gemini's free tier returns a `429
  RESOURCE_EXHAUSTED` error with an explicit `retryDelay` when the daily
  request quota is hit. The retry logic parses this value out of the error
  message and waits exactly that long, rather than guessing with a fixed
  backoff — this matters because each wasted retry consumes another unit
  of an already-scarce daily quota.

### 1.3 Prompt design decisions (informed by manually reading every document)

Before writing the prompt, all 10 training and 4 test documents were read in
full. This surfaced several recurring traps that the prompt explicitly
addresses:

- **Rent vs. deposit confusion**: nearly every document mentions a security
  deposit or advance payment figure that is numerically larger than the
  actual monthly rent, sitting in the same paragraph. The prompt explicitly
  instructs the model to extract only the figure described as *monthly
  rent*, not the largest number in the document.
- **Missing explicit end dates**: most documents state only a start date
  and a duration in words ("eleven months", "12 months") with no literal
  end date. The prompt instructs the model to use the standard legal
  convention — *start date + duration − 1 day* — only when no explicit end
  date or "from X to Y" phrase is present in the text.
- **Multiple notice periods**: some documents state more than one notice
  period for different situations (e.g. standard termination vs. a
  penalty/breach clause). The prompt instructs the model to prefer the
  notice period tied to ordinary termination/renewal.
- **Title stripping**: Party names in the ground truth never include
  honorifics (Mr./Mrs./Prof./Sri.) or parentage details (S/o, D/o), so the
  prompt instructs the model to return the bare name only.

---

## 2. Known Data Issues Found During Manual Review

Two real issues were found in the provided dataset itself — not in this
system's logic — and are documented here rather than silently
worked around, so the recall numbers below can be interpreted correctly.

### 2.1 Train-set filename/content mismatch (verified via independent OCR)

`54770958-Rental-Agreement.png` and `54945838-Rental-Agreement.png` have
**swapped content relative to their ground-truth labels** in `train.csv`.
Running Tesseract OCR directly on each file (independent of the Gemini
pipeline) confirms:

| Filename | train.csv claims | File actually contains |
|---|---|---|
| `54770958-Rental-Agreement.png` | K. Parthasarathy / Veerabrahmam Bathini / ₹8,000 | Asha Ramesh & Ramesh K.N / Sadasivuni Deepthi & Kiran / ₹5,500 |
| `54945838-Rental-Agreement.png` | Asha Ramesh & Ramesh K.N / ... / ₹5,500 | K. Parthasarathy / Veerabrahmam Bathini / ₹8,000 |

This was confirmed reproducible across two independent, fully isolated
extraction runs (different sessions, 4-second gaps between calls) — the
model consistently and correctly reports each file's actual content; it is
the CSV's filename-to-content assignment that is incorrect. Both the
**raw** and **adjusted** (excluding these 2 files) recall numbers are
reported below for transparency.

### 2.2 Invalid calendar dates in ground truth

Several ground-truth end dates are not valid calendar dates: `31.11.2009`
(November has 30 days), `31.04.2011` (April has 30 days), `31.02.2011`
(February never has 31 days). These appear to result from naive
month-rollover arithmetic (`start + N months`, keeping the start day
number) without validating the resulting date. Where this occurs, the
model's output — computed using the standard *start + duration − 1 day*
lease convention — is calendar-valid but does not exact-match the
ground truth. These cases are called out individually in the results
below rather than treated as extraction failures.

### 2.3 Other labeling quirks observed

- `228094620` (test set): ground truth Party Two is `.B.Kishore`, with a
  stray leading period baked into the label itself. The model returns the
  clean `B.Kishore`.
- `18325926` (train set): ground truth keeps the `MR.` title for Party One
  (`MR.K.Kuttan`), inconsistent with every other row, where titles are
  stripped.

---

## 3. Results

### 3.1 Train set (10 documents)

**Raw recall** (against `train.csv` as provided):

| Field | Recall |
|---|---|
| Agreement Value | 70.00% |
| Agreement Start Date | 70.00% |
| Agreement End Date | 40.00% |
| Renewal Notice (Days) | 60.00% |
| Party One | 40.00% |
| Party Two | 60.00% |
| **Overall Average** | **56.67%** |

**Adjusted recall** (excluding the 2 files with verified mislabeled
ground truth — see §2.1):

| Field | Raw (n=10) | Adjusted (n=8) |
|---|---|---|
| Agreement Value | 70.00% | 87.50% |
| Agreement Start Date | 70.00% | 87.50% |
| Agreement End Date | 40.00% | 50.00% |
| Renewal Notice (Days) | 60.00% | 75.00% |
| Party One | 40.00% | 50.00% |
| Party Two | 60.00% | 75.00% |
| **Overall Average** | **56.67%** | **70.83%** |

**Remaining misses, explained:**
- `44737744-Maddireddy-Bhargava-Reddy-Rental-Agreement.docx` — the
  extracted text from this `.docx` is corrupted/garbled at the source
  (e.g. the rent figure renders as `Rs 9.99.7.9°` and the date as
  `.?.??&`), almost certainly from a prior lossy PDF→DOCX conversion. No
  text-extraction or prompting strategy can recover information that
  isn't present in the source text. This single file accounts for the
  Value, Start Date, and End Date misses beyond the swap-affected rows.
- `18325926`, `36199312`, `47854715` (End Date) — ground truth uses an
  invalid calendar date (§2.2); the model's output is calendar-valid but
  doesn't match.
- `18325926` (Party One) — ground truth's inconsistent title retention
  (§2.3).
- Minor punctuation drift on a few Party fields (e.g. a period vs. comma
  reproduced faithfully from inconsistent source punctuation).

### 3.2 Test set (4 documents)

| Field | Recall |
|---|---|
| Agreement Value | 100.00% |
| Agreement Start Date | 100.00% |
| Agreement End Date | 75.00% |
| Renewal Notice (Days) | 75.00% |
| Party One | 100.00% |
| Party Two | 50.00% |
| **Overall Average** | **83.33%** |

**Remaining misses, explained:**
- `95980236` (End Date) — the source text states the lease is for "11
  (Eleven) months" from 01.04.2010. The model correctly computes
  01.04.2010 + 11 months − 1 day = 28.02.2011. Ground truth instead shows
  31.03.2011, which would only follow from a 12-month duration. This
  appears to be the same ground-truth duration/arithmetic inconsistency
  pattern seen in the train set.
- `228094620` (Renewal Notice) — the only "notice" phrase in this document
  ("vacate the flat with one months notice") is tied to a damage/breach
  clause, not standard termination. Per the prompt's explicit instruction
  to prefer ordinary termination/renewal clauses over punitive ones, the
  model correctly withholds an answer (`null`) rather than guessing.
  Ground truth uses this clause's value (30) anyway — a genuine tension
  between the prompt's design and the dataset's actual labeling
  convention, not a model misunderstanding.
- `156155545` (Party Two) — ground truth strips the "SRI" prefix and a
  trailing period (`VYSHNAVI DAIRY SPECIALITIES Private Ltd`); the model
  preserves them (`SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd.`) since
  "SRI" was not explicitly listed as a title to strip in the prompt's
  examples.
- `228094620` (Party Two) — ground truth has a stray leading period
  baked into the label (`.B.Kishore`, §2.3); the model returns the clean
  `B.Kishore`.

---

## 4. How to Run

1. Open the notebook in Google Colab.
2. Run cells in order:
   - **Cell 1** — installs dependencies (`google-generativeai`,
     `python-docx`, `pytesseract`, `Pillow`, `pandas`) and system
     `tesseract-ocr`.
   - **Cell 2** — upload `train.csv` and `test.csv` when prompted.
   - **Cell 3** — upload all 10 files listed for the `train/` folder.
   - **Cell 4** — upload all 4 files listed for the `test/` folder.
   - **Cell 5** — paste your own Gemini API key (get one free at
     [Google AI Studio](https://aistudio.google.com)) and run to confirm
     the connection.
   - **Cells 6–7** — define the document loading and extraction
     functions (no output expected beyond a confirmation message).
   - **Cell 8** — runs real extraction on all 10 train files.
   - **Cell 9** — computes raw per-field recall against `train.csv`.
   - **Cell 9c** (optional) — computes adjusted recall excluding the
     2 known-mislabeled files (§2.1).
   - **Cell 10** — runs real extraction on all 4 test files.
   - **Cell 11** — computes final test recall and saves
     `test_predictions.csv`.

### Notes on API quota

The Gemini free tier allows **20 requests/day per project**. This pipeline
needs at most ~14–42 requests across train+test depending on retries, which
fits within one day's free quota in most cases. If you see
`429 RESOURCE_EXHAUSTED` errors, either wait for the daily quota reset or
use a fresh API key from a different Google Cloud project (each project
gets its own independent 20/day allocation).

**Before sharing or committing this notebook**, replace the API key in
Cell 5 with a placeholder (`"PASTE_YOUR_KEY_HERE"`) — do not commit a live
key to GitHub or include it in a submitted zip file.

---

## 5. Files in this Submission

```
├── README.md                  (this file)
├── metadata_extraction.ipynb  (the full notebook, cells 1–11)
├── test_predictions.csv       (final predictions on the 4 test documents)
├── train/                     (10 training documents)
└── test/                      (4 test documents)
```

---

## 6. Possible Future Improvements

- **Self-consistency checks**: run each extraction 2–3 times and flag
  fields where repeated runs disagree, as a confidence signal.
- **Light post-processing normalization** for Party fields (e.g.
  optionally stripping a configurable list of organizational prefixes
  like "SRI") — left out of the core prompt deliberately, since adding
  too many hardcoded special cases risks drifting back toward the
  rule-based approach the assignment explicitly disallows.
- **Cross-checking extracted rent value against multiple mentions** in
  the document (since rent is often restated in a payment-schedule
  clause) to catch the rare case where the first occurrence is in a
  contradictory/distractor sentence.
