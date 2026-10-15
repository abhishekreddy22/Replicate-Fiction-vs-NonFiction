"""
main.py
=======
Master runner for:
  Qureshi et al., "A Simple Approach to Classify Fictional and Non-Fictional
  Genres", ACL Storytelling Workshop, 2019.

Execution order:
  1. Feature extraction
       • Brown Corpus  — gold tags  (324 docs)
       • Brown Corpus  — PTB  tags  (324 docs)
       • BNC Baby fic+aca — PTB tags (55 docs)
       • BNC Baby news   — PTB tags  (97 docs)
  2. Two-Stage RFECV (feature_selection.py)
       • Stage 1: 19 features → 6 features  (Brown gold, L1 LR, 10-fold CV)
       • Stage 2:  6 features → 2 features  (Brown PTB,  L1 LR, 10-fold CV)
  3. Table 4 Experiments (experiments.py)
       • Setup (a): Brown only,     10 × 60/40 splits
       • Setup (b): Brown + BNC,    10 × 60/40 splits
       • Setup (c): Train Brown → Test BNC
  4. Table 5 Experiments
       • BNC news classification rate under 4 training conditions
  5. Visualisations (visualize.py)
       • Figure 1: Scatter plot (key genre clusters)
       • Figure 2: POS ratio distributions
       • Figure 3: Punctuation ratio distributions
       • Figure 4: Low-level feature distributions
"""

import os
import pandas as pd

from data_handling import (
    preprocess_brown_corpus,
    preprocess_bnc_corpus,
    load_bnc_news,
)
from feature_selection import run_two_stage_rfecv, ACTIVE_FEATURES
from experiments import setup_a, setup_b, setup_c, table_5
from visualize import plot_figure1, plot_figure2, plot_figure3, plot_figure4

BNC_ROOT = "Corpus/baby_bnc/Texts"

if __name__ == "__main__":

    # ── 1. Feature Extraction ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 1 — Feature Extraction")
    print("="*70)

    print("\n[1/4] Brown Corpus — Gold tags …")
    df_brown_gold = preprocess_brown_corpus(tagset='brown')

    print("\n[2/4] Brown Corpus — PTB automatic tags …")
    df_brown_ptb = preprocess_brown_corpus(tagset='ptb')

    print("\n[3/4] BNC Baby (fic + aca) — PTB automatic tags …")
    df_bnc = preprocess_bnc_corpus(BNC_ROOT)

    print("\n[4/4] BNC Baby news — PTB automatic tags …")
    df_news = load_bnc_news(BNC_ROOT)

    print(f"\nData summary:")
    print(f"  Brown Gold : {df_brown_gold.shape[0]} docs, {df_brown_gold.shape[1]-2} features")
    print(f"  Brown PTB  : {df_brown_ptb.shape[0]} docs, {df_brown_ptb.shape[1]-2} features")
    print(f"  BNC fic+aca: {df_bnc.shape[0]} docs, {df_bnc.shape[1]-2} features")
    print(f"  BNC news   : {df_news.shape[0]} docs, {df_news.shape[1]-2} features")

    # ── 2. Table 4 Experiments ────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 2 — Table 4: Classification Experiments (ACTIVE_FEATURES only)")
    print("="*70)
    
    # Passing None for six_features and two_features as we paused RFECV 
    # and updated experiments.py to only use ACTIVE_FEATURES
    results_a = setup_a(df_brown_ptb, None, None)
    results_b = setup_b(df_brown_ptb, df_bnc, None, None)
    results_c = setup_c(df_brown_gold, df_brown_ptb, df_bnc, None, None)

    # ── 3. Table 5 Experiments ────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3 — Table 5: BNC News Generalisation (ACTIVE_FEATURES only)")
    print("="*70)

    table_5(df_brown_ptb, df_bnc, df_news, None, None)

    # ── 4. Visualisations ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4 — Generating Figures")
    print("="*70)

    # Figure 1 uses Brown PTB features (auto tags, as in paper's scatter)
    plot_figure1(df_brown_ptb, save_path='plots/figure1_scatter.png')
    plot_figure2(df_brown_ptb, save_path='plots/figure2_pos.png')
    plot_figure3(df_brown_ptb, save_path='plots/figure3_punct.png')
    plot_figure4(df_brown_ptb, save_path='plots/figure4_lowlevel.png')

    print("\n" + "="*70)
    print("DONE.")
    print("  Figures  : plots/figure1_scatter.png, plots/figure2_pos.png, "
          "plots/figure3_punct.png, plots/figure4_lowlevel.png")
    print("="*70)