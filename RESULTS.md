# Results and Analysis

This report summarizes the main findings from the Food.com dataset. For the ranking formula and data-processing rules, see the [README](README.md#methodology); for full calculations and plots, see the [ranking analysis notebook](notebooks/ranking_analysis.ipynb).

## 1. Dataset snapshot

- **231,637 recipes** and **1,132,367 user interactions**.
- **1,071,520 valid star ratings** (1–5); **60,847 zero-rating interactions** excluded from scoring.
- **226,590 rated recipes** included in the rankings; **5,047 unrated recipes** excluded.
- **91,698 recipes** have only one valid rating, illustrating how uneven rating evidence is.
- The global mean of individual valid ratings is **4.6615/5**.

Data validation found no duplicate recipe IDs or references to unknown recipes. Some recipe metadata, particularly preparation times, contain unusual values; these do not affect the ranking scores.

## 2. Comparing the rankings

The main analysis compares the naive mean-rating baseline with Bayesian-style shrinkage using **smoothing strength $m=10$**.

| Top-k | Shared recipes | Overlap |
|---:|---:|---:|
| 10 | 6 | 60% |
| 20 | 10 | 50% |
| 50 | 29 | 58% |
| 100 | 46 | 46% |

For the **top 20 recipes**, the median valid-rating count increases from **45** (naive ranking) to **79.5** (weighted ranking); the corresponding means are **57.7** and **121.6**. Thus, the weighted method selects a top-20 list supported by more rating evidence.

Some recipes also move substantially in rank. These changes should be interpreted carefully because many recipes have tied or nearly tied mean ratings.

## 3. Sensitivity to smoothing

The choice of $m$ changes the top-20 list. Relative to the naive top 20:

| Smoothing strength $m$ | Top-20 overlap |
|---:|---:|
| 1 | 100% |
| 5 | 75% |
| 10 | 50% |
| 20 | 30% |
| 50 | 10% |
| 100 | 5% |

A larger $m$ gives the global rating mean more influence. **$m=10$ is an illustrative setting, not an optimized choice.**

## 4. Interpretation and limitations

Accounting for rating counts **changes leaderboard composition and tends to favor recipes with more observed feedback**. However:

- More ratings do **not** necessarily mean higher recipe quality.
- User reviews can be subjective, sparse, and affected by selection bias.
- No held-out or future-rating evaluation establishes which method better predicts satisfaction.
- The ranking is **global**; tags, preparation time, and ingredient count are not ranking signals.

**Conclusion:** Bayesian-style shrinkage is a transparent, count-aware alternative to simple average ratings. These results demonstrate *how* rankings differ—not that one method is universally better.
