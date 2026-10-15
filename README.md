# Genre Classification Replication and Paragraph Ablation Study

This project replicates the stylometric genre classification approach and conducts paragraph ablation analysis.

## Overview

The codebase analyzes text data from two corpora:
- Brown Corpus (324 documents across 10 genres)
- BNC Baby Corpus (55 documents: fiction and academic genres)

The classifier distinguishes fictional genres from non-fictional genres using L1-regularized Logistic Regression across 19 stylometric features.

---

## Paragraph Ablation Analysis

The module [paragraph_analysis.py](paragraph_analysis.py) examines the effect of removing paragraphs from each document.

### Removal Strategies Tested
1. **Random Paragraphs**: Removes $k$ randomly sampled paragraphs.
2. **First Paragraphs**: Removes the first $k$ paragraphs from the start of the document.
3. **Last Paragraphs**: Removes the last $k$ paragraphs from the end of the document.

### Removal Percentages
- Evaluated at: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, and 90%.

### Evaluation Protocol
- 10 repeated 60/40 stratified train/test splits across 379 documents (Brown + BNC Baby).
- Evaluates four feature configurations:
  - All 19 Features
  - All Low-Level Features (11 features)
  - 6 Selected Features
  - 2 Selected Features

### Key Findings
- **High Stability (0% to 70% Removal)**: Model accuracy remains above 94% across all removal strategies. Stylometric ratios stay consistent throughout the text volume.
- **Degradation at Extreme Removal (80% to 90%)**: Accuracy drops when text volume becomes insufficient to represent stylistic distributions accurately.
- **Feature Set Robustness**: The full 19-feature set shows greater resilience to paragraph loss than the 2-feature subset at 90% reduction.

---

## Project Structure

- [main.py](main.py): Pipeline runner executing feature extraction, classification experiments, and figure generation.
- [paragraph_analysis.py](paragraph_analysis.py): Paragraph ablation experiments, caching, and plotting.
- [feature_selection.py](feature_selection.py): Feature definitions and two-stage RFECV logic.
- [experiments.py](experiments.py): Experimental setups and evaluation functions.
- [visualize.py](visualize.py): Visualization functions for distribution figures and scatter plots.
- [data_handling/](data_handling/): Corpus loaders and taggers for Brown and BNC Baby.

---

## Generated Plots

- [plots/paragraph_removal_accuracy.png](plots/paragraph_removal_accuracy.png): Four-panel plot of accuracy versus drop percentage across all feature sets.
- [plots/paragraph_removal_summary.png](plots/paragraph_removal_summary.png): Summary comparison for the 19-feature model with standard deviation error bars.
- [plots/figure1_scatter.png](plots/figure1_scatter.png): Genre cluster scatter plot.
- [plots/figure2_pos.png](plots/figure2_pos.png): Part-of-speech ratio distributions.
- [plots/figure3_punct.png](plots/figure3_punct.png): Punctuation ratio distributions.
- [plots/figure4_lowlevel.png](plots/figure4_lowlevel.png): Low-level feature distributions.
