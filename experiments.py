"""
experiments.py
==============
Replicates Tables 4 and 5 from:
  Qureshi et al., "A Simple Approach to Classify Fictional and Non-Fictional
  Genres", ACL Storytelling Workshop, 2019.

Table 4 — Classification accuracy across 4 feature sets and 3 setups:
  Feature sets:
    (A) All Low-Level  (11 features)
    (B) All 19 features
    (C) 6 Selected features
    (D) 2 Selected features

  Experimental setups:
    (a) Brown Corpus only       — 10 × random 60/40 stratified splits
    (b) Brown + BNC Baby        — 10 × random 60/40 stratified splits
    (c) Cross-corpus            — Train on all 324 Brown, test on 55 BNC

  Model: L1-regularised Logistic Regression (same as feature_selection.py)
  Metrics: Test Accuracy, F1 (non-fiction), F1 (fiction),
           Majority Baseline Accuracy, Accuracy Gain

Table 5 — % of BNC Baby news (97 docs) classified as non-fiction
  under 4 training conditions × 3 feature sets (19, 6, 2).
"""

import os
import warnings
# Suppress sklearn 1.8 deprecation of penalty= kwarg.
warnings.filterwarnings('ignore', message='.*penalty.*deprecated.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*penalty.*deprecated.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Inconsistent values.*penalty.*', category=UserWarning)
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODELS_DIR = 'models'
PLOTS_DIR  = 'plots'
CLASS_LABELS = ['Non-fiction', 'Fiction']

from feature_selection import (
    NF_CATEGORIES, LOW_LEVEL_FEATURES, ALL_FEATURES, ACTIVE_FEATURES
)

# ── label helper ────────────────────────────────────────────────────────────

# Non-fiction categories across both corpora
NF_CATEGORIES_BNC = {'aca'}          # BNC: aca = academic non-fiction
F_CATEGORIES_BNC  = {'fic'}          # BNC: fic = fiction


def _target_brown(df):
    """0 = non-fiction, 1 = fiction (Brown categories)."""
    return (df['Category'].map(
        lambda c: 0 if c in NF_CATEGORIES else 1
    )).values


def _target_bnc(df):
    """0 = non-fiction (aca), 1 = fiction (fic)."""
    return df['Category'].map({'aca': 0, 'fic': 1}).values


def _target_combined(df):
    """Unified target for Brown + BNC combined DataFrames."""
    def label(c):
        if c in NF_CATEGORIES or c == 'aca':
            return 0
        return 1
    return df['Category'].map(label).values


# ── model builder ───────────────────────────────────────────────────────────

def _lr():
    """L1-regularised Logistic Regression (paper's classifier).

    solver='liblinear' is optimal for small datasets with L1 regularization.
    """
    return LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        max_iter=1000,
        random_state=42,
    )


# ── metric helper ────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred, majority_class_size, n_total):
    """Return dict of accuracy, F1s, baseline, and accuracy gain."""
    acc      = accuracy_score(y_true, y_pred) * 100
    f1_nf    = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_f     = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    baseline = (majority_class_size / n_total) * 100
    gain     = ((acc - baseline) / (100 - baseline)) * 100 if baseline < 100 else 0.0
    return {
        'Accuracy':      acc,
        'F1_NF':         f1_nf,
        'F1_F':          f1_f,
        'Baseline':      baseline,
        'AccuracyGain':  gain,
    }


def _print_confusion_matrix(y_true, y_pred, label=''):
    """Print a text confusion matrix and save a matplotlib heatmap PNG.

    The PNG is saved to plots/cm_<safe_label>.png.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ── text output ──────────────────────────────────────────────────────
    title = f"Confusion Matrix — {label}" if label else "Confusion Matrix"
    print(f"\n  {title}")
    print(f"  {'':>20}  Predicted NF  Predicted Fiction")
    print(f"  {'Actual NF':>20}  {tn:>12}  {fp:>16}")
    print(f"  {'Actual Fiction':>20}  {fn:>12}  {tp:>16}")

    # ── heatmap ───────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    safe = label.replace(' ', '_').replace('—', '-').replace('(', '').replace(')', '').lower()
    save_path = os.path.join(PLOTS_DIR, f"cm_{safe}.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # axis labels
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_LABELS, fontsize=10)
    ax.set_yticklabels(CLASS_LABELS, fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=11, labelpad=8)
    ax.set_ylabel('Actual label', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, pad=12)

    # count annotations — white text on dark cells, black on light cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix plot saved → {save_path}")


def _save_model(model, scaler, filename):
    """Save model and scaler to the models/ directory using joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f"  Model saved → {path}")


# ── feature-set definitions ──────────────────────────────────────────────────

def build_feature_sets(six_features, two_features):
    """Return ordered dict of {label: feature_list} for Table 4."""
    # We only return the ACTIVE_FEATURES in custom analysis mode.
    # Otherwise, this could return the full set. For now, strictly use ACTIVE_FEATURES.
    return {
        'Active Features': ACTIVE_FEATURES
    }


# ── Setup (a) & (b): 10 × 60/40 splits ─────────────────────────────────────

def _run_repeated_splits(X, y, n_splits=10, test_size=0.4):
    """Run n_splits random 60/40 stratified splits.

    Returns mean ± std metrics and the model from the last split.
    """
    results = []
    majority_size = max(np.sum(y == 0), np.sum(y == 1))
    last_model, last_scaler, last_y_te, last_y_pred = None, None, None, None

    for seed in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        model = _lr()
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        results.append(_metrics(y_te, y_pred, majority_size, len(y)))
        last_model, last_scaler = model, sc
        last_y_te, last_y_pred  = y_te, y_pred

    keys = ['Accuracy', 'F1_NF', 'F1_F', 'Baseline', 'AccuracyGain']
    stats = {
        k: (np.mean([r[k] for r in results]),
            np.std([r[k]  for r in results]))
        for k in keys
    }
    return stats, last_model, last_scaler, last_y_te, last_y_pred


def setup_a(df_brown, six_features, two_features):
    """Table 4 Setup (a): Brown Corpus, 10 × random 60/40 stratified splits."""
    print("\n" + "="*70)
    print("TABLE 4 — Setup (a): Brown Corpus  [10 × 60/40 stratified splits]")
    print("="*70)

    feature_sets = build_feature_sets(six_features, two_features)
    y = _target_brown(df_brown)
    rows = []

    for label, feats in feature_sets.items():
        X = df_brown[feats].values
        stats, model, scaler, y_te, y_pred = _run_repeated_splits(X, y)
        rows.append(_format_row(label, stats))
        # Confusion matrix uses the last split as a representative sample.
        _print_confusion_matrix(y_te, y_pred, label=f"Setup (a) — {label}")
        safe_label = label.replace(' ', '_').lower()
        _save_model(model, scaler, f"setup_a_{safe_label}.joblib")

    _print_table(rows)
    return rows


def setup_b(df_brown, df_bnc, six_features, two_features):
    """Table 4 Setup (b): Brown + BNC Baby, 10 × random 60/40 stratified splits."""
    print("\n" + "="*70)
    print("TABLE 4 — Setup (b): Brown + BNC Baby  [10 × 60/40 stratified splits]")
    print("="*70)

    # Align BNC category labels to match Brown target encoding:
    # aca → NF, fic → F
    df_bnc_aligned = df_bnc.copy()
    df_bnc_aligned['Category'] = df_bnc_aligned['Category'].map(
        {'aca': 'government', 'fic': 'fiction'}
    )

    combined = pd.concat([df_brown, df_bnc_aligned], ignore_index=True)
    feature_sets = build_feature_sets(six_features, two_features)
    y = _target_brown(combined)
    rows = []

    for label, feats in feature_sets.items():
        X = combined[feats].values
        stats, model, scaler, y_te, y_pred = _run_repeated_splits(X, y)
        rows.append(_format_row(label, stats))
        # Confusion matrix uses the last split as a representative sample.
        _print_confusion_matrix(y_te, y_pred, label=f"Setup (b) — {label}")
        safe_label = label.replace(' ', '_').lower()
        _save_model(model, scaler, f"setup_b_{safe_label}.joblib")

    _print_table(rows)
    return rows


def setup_c(df_brown_gold, df_brown_ptb, df_bnc, six_features, two_features):
    """Table 4 Setup (c): Train on Brown GOLD tags, Test on BNC Baby PTB tags.

    This is the critical cross-corpus experiment in the paper.
    The training uses Brown GOLD-standard annotations (manually annotated),
    while the test set (BNC) uses AUTOMATIC PTB tags. This distributional
    mismatch causes the 19-feature model to collapse to ~52% (worse than
    baseline), while the 2-feature model maintains ~100% accuracy — the
    paper's central finding.
    """
    print("\n" + "="*70)
    print("TABLE 4 — Setup (c): Train on Brown GOLD → Test on BNC Baby PTB")
    print("  (Paper's key experiment: gold tags train, auto tags test)")
    print("="*70)

    feature_sets = build_feature_sets(six_features, two_features)
    y_train = _target_brown(df_brown_gold)   # labels from gold-tag Brown DF
    y_test  = _target_bnc(df_bnc)
    majority_size = max(np.sum(y_test == 0), np.sum(y_test == 1))
    rows = []

    for label, feats in feature_sets.items():
        # Train features extracted from Brown with GOLD tags
        X_train = df_brown_gold[feats].values
        # Test features extracted from BNC with automatic PTB tags
        X_test  = df_bnc[feats].values

        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s  = sc.transform(X_test)

        model = _lr()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        m = _metrics(y_test, y_pred, majority_size, len(y_test))
        rows.append(_format_row_single(label, m))
        _print_confusion_matrix(y_test, y_pred, label=f"Setup (c) — {label}")
        safe_label = label.replace(' ', '_').lower()
        _save_model(model, sc, f"setup_c_{safe_label}.joblib")

    _print_table_single(rows)
    return rows


# ── Table 5: News generalisation ─────────────────────────────────────────────

def table_5(df_brown, df_bnc, df_news, six_features, two_features):
    """Table 5: % of BNC news classified as non-fiction under 4 training sets.

    Training conditions:
      T1: Brown only (324 docs)
      T2: Brown without news subgenre (280 docs)
      T3: Brown + BNC Baby fic+aca (379 docs)
      T4: Brown + BNC Baby fic+aca+news (476 docs)

    Feature sets tested: 19 features, 6 features, 2 features.
    """
    print("\n" + "="*70)
    print("TABLE 5 — % BNC News classified as Non-fiction")
    print("="*70)

    feature_sets = {
        'Active Features': ACTIVE_FEATURES
    }

    # Build the 4 training DataFrames
    # T2: Brown without its 'news' subgenre
    df_brown_no_news = df_brown[df_brown['Category'] != 'news'].copy()

    # Align BNC aca/fic to Brown's encoding so target function works
    df_bnc_aligned = df_bnc.copy()
    df_bnc_aligned['Category'] = df_bnc_aligned['Category'].map(
        {'aca': 'government', 'fic': 'fiction'}
    )
    df_bnc_news_aligned = df_news.copy()
    df_bnc_news_aligned['Category'] = 'news'  # treated as non-fiction

    training_sets = {
        'T1: Brown (324)':              df_brown,
        'T2: Brown−news (280)':         df_brown_no_news,
        'T3: Brown + BNC (379)':        pd.concat([df_brown, df_bnc_aligned], ignore_index=True),
        'T4: Brown + BNC + news (476)': pd.concat([df_brown, df_bnc_aligned, df_bnc_news_aligned],
                                                   ignore_index=True),
    }

    # Table header
    feat_labels = list(feature_sets.keys())
    header = f"{'Training set':<30}" + "".join(f"  {fl:<18}" for fl in feat_labels)
    print(header)
    print("-" * len(header))

    for t_label, df_train in training_sets.items():
        row_str = f"{t_label:<30}"
        y_train = _target_brown(df_train)

        for feat_label, feats in feature_sets.items():
            X_train = df_train[feats].values
            X_news  = df_news[feats].values

            sc = StandardScaler()
            X_train_s = sc.fit_transform(X_train)
            X_news_s  = sc.transform(X_news)

            model = _lr()
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_news_s)

            pct_nf = (np.sum(y_pred == 0) / len(y_pred)) * 100
            row_str += f"  {pct_nf:>6.1f}%            "

            # Save model for each training condition and feature set.
            safe_t = t_label.split(':')[0].strip().lower().replace(' ', '_')
            safe_f = feat_label.replace(' ', '_').lower()
            _save_model(model, sc, f"table5_{safe_t}_{safe_f}.joblib")

        print(row_str)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_row(label, stats):
    """Format a row for the repeated-split table (mean ± std)."""
    acc  = stats['Accuracy']
    f1nf = stats['F1_NF']
    f1f  = stats['F1_F']
    bl   = stats['Baseline']
    ag   = stats['AccuracyGain']
    return {
        'Feature Set':    label,
        'Accuracy':       f"{acc[0]:.2f} ± {acc[1]:.2f}",
        'F1 (NF)':        f"{f1nf[0]:.4f} ± {f1nf[1]:.4f}",
        'F1 (F)':         f"{f1f[0]:.4f} ± {f1f[1]:.4f}",
        'Baseline':       f"{bl[0]:.2f}",
        'Acc. Gain':      f"{ag[0]:.2f} ± {ag[1]:.2f}",
    }


def _format_row_single(label, m):
    """Format a row for a single-split table."""
    return {
        'Feature Set':    label,
        'Accuracy':       f"{m['Accuracy']:.2f}",
        'F1 (NF)':        f"{m['F1_NF']:.4f}",
        'F1 (F)':         f"{m['F1_F']:.4f}",
        'Baseline':       f"{m['Baseline']:.2f}",
        'Acc. Gain':      f"{m['AccuracyGain']:.2f}",
    }


def _print_table(rows):
    cols = ['Feature Set', 'Accuracy', 'F1 (NF)', 'F1 (F)', 'Baseline', 'Acc. Gain']
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in cols))


def _print_table_single(rows):
    _print_table(rows)
