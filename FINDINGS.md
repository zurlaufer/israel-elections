# Box-Level Election Forecasting: Findings

Elections 21-25 (Sep 2019 - Nov 2022), box-level results, CBS 2022 census.

## Pipeline

| File | Produces | Purpose |
|---|---|---|
| `src/download.py` | `data/raw/*.json` | Pull raw election + census files |
| `src/normalize.py` | `data/processed/results_long.parquet` | Box x party long table, unified schema |
| `src/party_map.py` | `data/processed/parties.csv` | Party name/letter registry per election |
| `src/families.py` | — | Party→family crosswalk + national family-share audit (single source of truth for family assignment) |
| `src/census.py` | `data/processed/locality_demographics.csv` | Locality profiles from 2022 census (1,185 localities, 96% of election localities) |
| `src/match.py` | `box_panel.parquet` (5-elec), `box_panel_train.parquet` (21-24 + e25 joined) | Stable box panel + box-identity analysis |
| `src/representative.py` | `box_alignments.parquet` | Leave-one-out validation of representative-box selection |
| `src/predict.py` | — | Box-level vote-share and turnout prediction for election 25 |
| `src/forecast.py` | `output/e25_predictions.parquet` | Full-scope election-25 forecast for all boxes + transition/structural-break analysis |

Data pipeline output is reproducible end-to-end: `python src/match.py` -> `src/representative.py` -> `src/predict.py`.

## 1. Box identity: `ברזל` serials are NOT stable

Hypothesis: the physical ballot-box serial (`ברזל`) is a stable box identifier across elections. **Refuted.**

Serials are renumbered every election. For the same `(locality, kalpi)`, the serial changed (e.g. Abu Ghosh kalpi 1.0: serial 733 in e22, 736 in e23, 841 in e24, 806 in e25). Within a locality serials are just a sequential index over boxes in that election's box set, so any change to the box configuration shifts all serials.

| Pair | Matched boxes | Serial agreement |
|---|---|---|
| 21-22 | — | n/a (e21 has no serials) |
| 22-23 | 10,047 | 0.4% |
| 23-24 | 4,614 | 0.1% |
| 24-25 | 7,542 | 9.0% |
| 22-25 | 5,116 | 0.1% |

Serials can therefore not be used as a panel key.

## 2. Kalpi is the usable key, but election 24 renumbered boxes

`(locality_code, kalpi)` is unique within each election and mostly stable across adjacent elections 21-23, but election 24 introduced wide sub-box renumbering:

| Pair | Match rate (of earlier election) |
|---|---|
| 21-22 | 90.9% |
| 22-23 | 92.2% |
| 23-24 | 41.3% |
| 24-25 | 58.3% |
| 22-25 | 46.9% |

**Stable panels built** (non-special boxes only, voters > 0):
- All 5 elections: **3,023 boxes** in 943 localities.
- Elections 21-24 (training) with e25 joined where the box survives: **3,639 boxes**, 969 localities, 3,023 of which also appear in e25.

## 3. Representative boxes generalize out-of-sample

Metric: *alignment* = vote-vector overlap between a box and its locality total in the same election (`sum_p min(box_share_p, loc_share_p)`), computed per election within that election's party set (no cross-election party alignment needed).

For each held-out election, per locality pick the box with the best mean training alignment, then read its alignment in the held-out election. Compare vs a random box and the worst box (localities with >= 3 panel boxes and >= 2 with held-out data).

| Held out | Localities | Persistence | best | random | worst |
|---|---|---|---|---|---|
| 21 | 103 | 0.907 | 0.305 | 0.281 | 0.237 |
| 22 | 103 | 0.918 | 0.355 | 0.328 | 0.289 |
| 23 | 104 | 0.901 | 0.423 | 0.399 | 0.353 |
| 24 | 113 | 0.755 | 0.271 | 0.253 | 0.221 |
| 25 | 106 | 0.876 | 0.293 | 0.272 | 0.256 |
| **avg** | | **0.871** | **0.330** | **0.307** | **0.271** |

- Best-in-train beats random in every held-out election (avg +0.023).
- Within-locality persistence correlation of train vs held-out alignment is 0.87 on average; it drops in the renumbering year (e24, 0.755), where box-level identity is noisier.
- Representative boxes are also more likely to survive renumbering into the next election than random boxes (e.g. e25: 87% vs 84%).

## 4. Party families must be assigned by party, not by ballot letter

Ballot letters are reused across elections for different parties, so any letter-keyed family map silently changes meaning between elections. Two concrete failures in the original mapping:

| letter | e23 | e24 | e25 |
|---|---|---|---|
| פה | Blue & White 26.34% | Yesh Atid 13.94% | Yesh Atid 17.88% |
| כן | — | Blue & White 6.66% | National Unity 8.91% |
| טב / ב | Yamina 5.01% | Yamina 5.87% | — |

The old map listed only פה under `Center`, so Blue & White's e24 list (כן) and National Unity's e25 list (כן) both fell into the `Other` plug. `Other` was computed as `total - named`, so it absorbed the whole error and swung 5.8% → 23.6% between e23 and e24 — a change in *ballot composition* (30 → 39 letters), not voters switching parties.

`src/families.py` now derives letters from party identity via `parties.csv`, so the map is generated and cannot drift. National family shares after the fix:

| Family | e21 | e22 | e23 | e24 | e25 |
|---|---|---|---|---|---|
| Likud | 26.29 | 24.92 | 29.22 | 24.17 | 23.40 |
| Center | 29.53 | 25.68 | 26.34 | 25.29 | 26.79 |
| Labor | 4.42 | 4.80 | 5.71 | 5.88 | 3.54 |
| Shas | 6.10 | 7.52 | 7.73 | 7.32 | 8.40 |
| UTJ | 5.92 | 6.19 | 6.12 | 5.88 | 6.11 |
| YB | 4.14 | 7.12 | 5.88 | 5.54 | 4.36 |
| Meretz | 3.64 | 4.30 | 0.00 | 4.56 | 3.17 |
| Arab | 8.12 | 11.04 | 13.23 | 9.08 | 11.34 |
| Religious Right | 6.80 | 7.53 | 5.01 | 10.85 | 11.51 |
| *residual Other* | *5.04* | *0.90* | *0.76* | *1.43* | *1.38* |

- `Center` is now comparable across all five elections (was 25.97/25.66/26.34/13.94/17.88).
- `Other` is now a genuine 0.8–5% residual instead of a 5.8–23.6% artifact, and is excluded from all headline MAE figures below (reported separately).
- `Religious Right` (Yamina, Religious Zionism, New Right, Otzma Yehudit, URP, Jewish Home) was previously split between `Other` and nothing; it is a coherent 5–11.5% bloc and is now its own family.
- Meretz is genuinely absent in e23 (it ran inside Labor's list that year), so its 0.00 is real, not a mapping failure.

Two `party_map.py` resolution bugs fixed: `המחנה הדמוקרטי` (e22 Meretz, 4.30%) matched no pattern and fell through unresolved, and the loose `הממלכתי` pattern mis-assigned an e21 Kach list (0.01%) to National Unity.

## 5. Box history beats locality history for prediction

Party lists were grouped into 9 families stable across all 5 elections by ballot letter (Likud, Center = Blue&White/Yesh Atid, Labor, Shas, UTJ, Yisrael Beiteinu, Meretz, Arab lists, Other). Election-25 shares were predicted for the 3,023 panel boxes using only pre-election information, evaluated as vote-weighted mean per-family MAE over the 9 mapped families (residual excluded).

| Method | MAE | (previous letter-keyed mapping) |
|---|---|---|
| box last election (24) | **0.0212** | 0.0233 |
| box median 21-24 | 0.0216 | 0.0330 |
| blend 0.5*box + 0.5*locality | 0.0272 | 0.0375 |
| locality median 21-24 | 0.0390 | 0.0479 |
| national average | 0.0791 | 0.0740 |
| equal shares | 0.1156 | 0.1146 |

- A box's own most recent result is the strongest predictor; any use of the box's own history beats the locality median.
- Box identity beats the locality median in **68.7%** of boxes (mean L1 distance 0.202 vs 0.350).
- Per-family, box history helps most for Likud (0.040 vs 0.065), Center (0.027 vs 0.071), Shas (0.015 vs 0.036), UTJ (0.008 vs 0.031), Arab lists (0.007 vs 0.021). **Religious Right is the hardest family for every method** (0.052 vs 0.058) — it nearly doubled between e23 and e25, so a box's own last result is a poor guide.
- Residual `Other` MAE is 0.0093 for both box and locality medians.

**Turnout** (voters/eligible): box median MAE 0.027 vs locality median 0.054 — box history roughly halves turnout error.

## 5a. The panel is a small and demographically skewed sample

The numbers above are computed on the 3,023-box panel, which is **not representative** of Israel. It holds only 23.5% of e25 valid votes, and the missing votes are concentrated in exactly the distinctive localities:

| Family | national | in panel | off panel | panel − national |
|---|---|---|---|---|
| Likud | 23.40 | 26.69 | 22.39 | **+3.29** |
| Center | 26.79 | 28.98 | 26.11 | +2.19 |
| UTJ | 6.11 | 3.53 | 6.90 | **−2.58** |
| Arab | 11.34 | 4.71 | 13.37 | **−6.62** |
| Religious Right | 11.51 | 13.44 | 10.92 | +1.93 |

- The panel contains **less than half the Arab vote** (4.71% vs 11.34%) and ~58% of the UTJ vote is off-panel. Requiring an identical `(locality, kalpi)` in all five elections drops most Arab and Haredi towns, because their box numbering changed.
- So §5's panel MAE is measured on an easier, more Likud-leaning sample than the country. The trustworthy figure is §6's, which is computed on all 11,707 boxes.
- The largest off-panel localities are small yeshuvim (~0.14% of votes each); the skew is driven by a long tail of renumbered boxes plus the absence of the big Arab towns.

## 6. Full-scope forecast: the 24->25 transition is a structural break

`src/forecast.py` scales the forecast to **all** 11,707 non-special election-25 boxes with votes (not just the 3,023 panel boxes). It also fits a weighted linear (ridge) model on 70 features (own/locality family shares over lags 1-3, turnout, eligible, census, box counts) over 33,295 box-transitions from elections 21-24.

Best forecast using only pre-election information:

| Method | all | matched (own 24 exists) | unmatched |
|---|---|---|---|
| **own 24, else locality 24** | **0.0283** | **0.0205** | **0.0383** |
| box own 24 (matched only) | 0.0205 | 0.0205 | — |
| locality 24 | 0.0392 | 0.0399 | 0.0383 |
| ridge model (70 features) | 0.0420 | 0.0351 | 0.0508 |
| national average | 0.0945 | 0.0944 | 0.0946 |
| equal shares | 0.1215 | 0.1210 | 0.1222 |

- 6,868 boxes (59%) match to an election-24 box; their forecast is the box's own 24 share (MAE 0.0205). The remaining 4,839 rely on the locality-24 share (MAE 0.0383).
- The ridge model **loses to the simple baselines** even in-sample it was near-perfect. Cause: each election transition has a different national swing signature, so nothing learned from 21-24 transfers to 24->25:

| Transition | Likud | Center | Labor | Shas | UTJ | YB | Meretz | Arab | Rel. Right | Other |
|---|---|---|---|---|---|---|---|---|---|---|
| 21->22 | -0.008 | -0.032 | +0.005 | +0.014 | +0.003 | +0.031 | +0.005 | +0.014 | +0.009 | -0.041 |
| 22->23 | +0.047 | +0.009 | +0.011 | +0.003 | 0.000 | -0.012 | -0.043 | +0.012 | -0.024 | -0.002 |
| 23->24 | -0.061 | -0.025 | -0.001 | -0.006 | -0.006 | -0.006 | +0.048 | -0.009 | +0.058 | +0.008 |
| 24->25 | -0.009 | +0.024 | -0.022 | +0.008 | +0.002 | -0.012 | -0.013 | +0.011 | +0.009 | -0.000 |

- The old `+0.184 Other` cell in the 23->24 row was a mapping artifact. With the fix, 23->24 is explained by real movement: Religious Right +0.058 (Yamina plus Religious Zionism on the ballot) and Meretz +0.048 (Meretz returned as a separate list after sitting inside Labor in e23). The residual `Other` row is now flat (+0.008), as it should be.
- **A national-swing correction does not help.** Giving the forecast the actual 24->25 national swing (an oracle a poll could approximate) leaves matched-box MAE unchanged at 0.0205 → 0.0209, i.e. marginally *worse*. Box-level changes that year were largely idiosyncratic, not a uniform swing.
- Conclusion: box share *levels* are highly persistent (matched-box MAE 0.021), but year-to-year *changes* are not learnable from the three prior transitions — forecasting an election needs external info (polls), not just box history.

## 7. Census locality profiles

`src/census.py` builds per-locality profiles by aggregating statistical areas weighted by population (falling back to locality-level rows for small localities): population, religion shares, religiosity scale (1=secular ... 4=Haredi), median age, median wage, academic share, homeownership. Sanity checks hold (Bnei Brak: religiosity 4.0, median age 19; Abu Ghosh: 100% Muslim).

## Limitations

- **The panel is unrepresentative** (§5a): it holds 23.5% of e25 valid votes and under-counts the Arab vote by 6.62pp and UTJ by 2.58pp. Every panel-only number (§3, §5) describes a more Likud-leaning Israel than the real one. Only §6's all-box figures describe the country.
- 41% of election-25 boxes do not match to election 24 (59% do); for those the forecast degrades to locality-level information (MAE 0.0383 vs 0.0205). Errors concentrate in the Arab and Haredi towns that are off-panel (§5a).
- Selected representative boxes skew slightly more Jewish and come from smaller localities than the average panel box.
- Election-24 box renumbering (41% match) is the main source of panel attrition and reduces predictive persistence for that transition.
- Family grouping still collapses party-level granularity (Blue & White, Yesh Atid and National Unity are one `Center` family). The Likud-Labor joint list (מחל) is counted entirely as Likud in all five elections, so `Labor` is understated everywhere by roughly the joint list's minor-party partners.
- The transition table and structural-break conclusion rest on a single held-out election (24->25) and 33,295 pooled box-transitions. Leave-one-election-out CV is the right next step.
- The ridge model's 70 features are fit on pooled 22-24 transitions; the 23->24 ballot-composition change (30 -> 39 letters) sits inside the training window, so some features encode composition rather than preference.
