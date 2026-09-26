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
| `src/cv.py` | `cv_summary.csv` | Leave-one-election-out validation of the forecasting method over the 21->22, 22->23, 23->24 and 24->25 transitions |
| `src/swing.py` | — | Tests whether a *differential* (stratum) swing beats a *uniform* national swing |

Data pipeline output is reproducible end-to-end: `python src/match.py` -> `src/representative.py` -> `src/predict.py`.
Validation (no outputs of their own beyond `cv_summary.csv`): `python src/forecast.py`, `python src/cv.py`, `python src/swing.py`.

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

**Caveat, resolved in §8:** the "structural break" framing was based on a single held-out transition. Leave-one-election-out CV shows the method ordering is stable, but that 24->25 is *not* uniquely broken — 23->24 is harder for every method. The swing-signature explanation above still stands, and §9 pushes on it further.

## 7. Census locality profiles

`src/census.py` builds per-locality profiles by aggregating statistical areas weighted by population (falling back to locality-level rows for small localities): population, religion shares, religiosity scale (1=secular ... 4=Haredi), median age, median wage, academic share, homeownership. Sanity checks hold (Bnei Brak: religiosity 4.0, median age 19; Abu Ghosh: 100% Muslim).

## 8. Leave-one-election-out CV: the §6 conclusions hold, but the sample is thin

§6 rested on a single held-out transition. `src/cv.py` repeats the whole §6 comparison for each of the four possible held-out elections, refitting the ridge model only on elections earlier than the target (so no future information leaks). Metric is unchanged: vote-weighted mean per-family MAE over the 9 mapped families, on **all** boxes with votes in the held-out election.

| Method | 21->22 | 22->23 | 23->24 | 24->25 | avg |
|---|---|---|---|---|---|
| box own prev (matched only) | 0.0207 | 0.0232 | 0.0331 | 0.0205 | **0.0244** |
| own prev, else locality prev | 0.0223 | 0.0256 | 0.0425 | 0.0283 | **0.0297** |
| locality prev | 0.0402 | 0.0458 | 0.0480 | 0.0392 | 0.0433 |
| ridge model (70 features) | — | 0.0381 | 0.0660 | 0.0420 | 0.0487 |
| national average | 0.0933 | 0.0963 | 0.0918 | 0.0945 | 0.0940 |
| equal shares | 0.1197 | 0.1328 | 0.1154 | 0.1215 | 0.1224 |

The `box own prev (matched only)` row is computed only on boxes that have a prior-election match — 92.8% of them in 21->22, 91.1% in 22->23, 33.9% in 23->24 and 58.7% in 24->25 — so it is **not** comparable to the other rows' `avg`. It is the achievable floor if box identity were perfect, and the gap to `own prev, else locality prev` is the cost of unmatched boxes.

- **The ordering in §6 was not a 24->25 accident.** The simple persistence method beats `locality prev` in 4/4 held-out elections, and `box own prev` beats it in 4/4 on matched boxes. Levels are persistent in every transition, not just the last one.
- **The ridge model loses in all 3 elections where it can be fit** (0.0381 / 0.0660 / 0.0420 vs 0.0256 / 0.0425 / 0.0283). It is not that e24->e25 was a one-off structural break — *no* pooled-transition model beats simply carrying the last result forward, in any held-out year.
- 23->24 is the hardest transition for all four informative methods (renumbering year, §2) — 0.0425 for the production method vs 0.0223-0.0283 elsewhere. Only the two constant-share baselines peak in 22->23 instead. So the "structural break" in §6 is better read as **renumbering attrition plus a genuinely large Likud swing (0.083 family MAE)**, not as a unique 24->25 event.
- Matched vs unmatched is stable across transitions: matched 0.0207-0.0331, unmatched 0.0384-0.0503. The ~2x penalty for unmatched boxes is a persistent property of the data, not a one-year artifact.

Per-family MAE of the production method, by transition:

| Family | 21->22 | 22->23 | 23->24 | 24->25 |
|---|---|---|---|---|
| Likud | 0.0252 | 0.0504 | 0.0827 | 0.0415 |
| Center | 0.0403 | 0.0340 | 0.0663 | 0.0557 |
| Labor | 0.0169 | 0.0237 | 0.0203 | 0.0229 |
| Shas | 0.0201 | 0.0129 | 0.0321 | 0.0271 |
| UTJ | 0.0106 | 0.0102 | 0.0344 | 0.0234 |
| YB | 0.0324 | 0.0162 | 0.0252 | 0.0203 |
| Meretz | 0.0231 | 0.0420 | 0.0455 | 0.0176 |
| Arab | 0.0159 | 0.0143 | 0.0184 | 0.0156 |
| Religious Right | 0.0162 | 0.0270 | 0.0578 | 0.0307 |

- The two large blocs dominate: Likud is the hardest family in 2/4 transitions and Center in the other 2/4. They carry most of the error because that is where swings are largest, not because their box-level tracking is bad.
- Arab-list MAE is both the most stable (range 0.0156-0.0184, vs 0.024-0.034 for the other families) and in the lowest tier alongside UTJ and Labor, so the under-representation noted in §5a costs coverage, not accuracy, on the boxes that *are* covered.
- Religious Right stays among the hardest (0.0578 in 23->24, third only to Likud and Center), consistent with §5's finding that it is the least predictable family for every method — its near-doubling between e23 and e25 is not something a box's own last result can anticipate.

## 9. A differential swing does not beat a uniform national swing

§6 showed an *oracle* national swing does not help. `src/swing.py` asks the stronger question: is the swing **differential by demographic stratum**? Strata are `religiosity (1-4) x locality size (small/medium/large)`, 14 cells. For each held-out transition, stratum mean share-changes are estimated **only from earlier transitions** and used to correct the box's own previous share; strata with < 200 training box-transitions fall back to the uniform mean. Target is the actual change in box share, vote-weighted.

| transition | train pairs | usable strata | vote coverage | uniform | differential | delta |
|---|---|---|---|---|---|---|
| 22->23 | 1 | 6/14 | 0.975 | **0.0323** | 0.0329 | +0.0006 |
| 23->24 | 2 | 6/14 | 0.945 | **0.0408** | 0.0411 | +0.0003 |
| 24->25 | 3 | 7/14 | 0.975 | **0.0225** | 0.0225 | -0.0000 |
| **avg** | | | | **0.0319** | 0.0322 | **+0.0003** |

- **Differential swing is marginally *worse* than no correction at all.** Even with 94-98% of votes covered by a usable stratum, stratifying on the most obvious demographic axis buys nothing.
- The reason is visible in an exact ANOVA decomposition of the pooled share-change variance (stratum means from training pairs only): strata explain **10.5%** going into e22, **3.4%** into e23, **1.6%** into e24. Real box-level movement is overwhelmingly idiosyncratic *within* a demographic stratum, and the stratum signal decays as more history is added.
- Per-family, the only family that clearly benefits is **Arab lists** (0.0182 -> 0.0155, -0.0027); every other family's delta lies between -0.0001 and +0.0032, i.e. noise or harm. The Arab gain is the one place where a demographic tilt is real — the Arab vote's reaction differs sharply from the national one — but it is small relative to the noise.
- Combined with §6's oracle-swing null result, the conclusion is firm: **box-level change is not predictable from box history, demography, or a national swing.** Carrying levels forward gets you MAE ~0.030; any model of *change* makes it worse. Forecasting an election requires external information (polls), which is exactly the quantity this dataset lacks.


## Limitations

- **The panel is unrepresentative** (§5a): it holds 23.5% of e25 valid votes and under-counts the Arab vote by 6.62pp and UTJ by 2.58pp. Every panel-only number (§3, §5) describes a more Likud-leaning Israel than the real one. Only §6's all-box figures describe the country.
- 41% of election-25 boxes do not match to election 24 (59% do); for those the forecast degrades to locality-level information (MAE 0.0383 vs 0.0205). Errors concentrate in the Arab and Haredi towns that are off-panel (§5a). The ~2x matched/unmatched penalty is stable across all four held-out transitions (§8), so it is a data property, not a fixable modeling gap.
- Selected representative boxes skew slightly more Jewish and come from smaller localities than the average panel box.
- Election-24 box renumbering (41% match) is the main source of panel attrition and reduces predictive persistence for that transition.
- Family grouping still collapses party-level granularity (Blue & White, Yesh Atid and National Unity are one `Center` family). The Likud-Labor joint list (מחל) is counted entirely as Likud in all five elections, so `Labor` is understated everywhere by roughly the joint list's minor-party partners.
- ~~The transition table and structural-break conclusion rest on a single held-out election.~~ **Resolved in §8**: leave-one-election-out CV over four transitions reproduces the ordering, and shows 24->25 was not a unique structural break (23->24 is harder). What remains unresolved is *why* the ridge model fails everywhere; the transition table shows each pair has a different national swing signature, but four transitions is not enough to learn a mapping from swing signature to box response.
- The ridge model's 70 features are fit on pooled transitions; the 23->24 ballot-composition change (30 -> 39 letters) sits inside the training window, so some features encode composition rather than preference. In the 21->22 fold the model cannot be fit at all (no prior transitions), so its CV average rests on 3 of 4 folds.
- §9's strata use only religiosity and locality size. Party-level, age, or income-based strata were not tested; the Arab-list gain suggests a more targeted axis might do better than 14 coarse cells.
- The `Other` residual is excluded from all headline MAE figures and is not itself forecast. It is genuine in size (0.8-5%) but still partly a mapping artifact, so no claim is made about it.

