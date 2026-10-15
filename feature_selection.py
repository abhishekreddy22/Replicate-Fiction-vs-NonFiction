"""
feature_selection.py
====================
Two-stage Recursive Feature Elimination with Cross-Validation (RFECV) exactly
as described in Section 4 of:
  Qureshi et al., "A Simple Approach to Classify Fictional and Non-Fictional
  Genres", ACL Storytelling Workshop, 2019.

Stage 1: RFECV on Brown GOLD tags → 19 features → 6 features
Stage 2: RFECV on Brown PTB  tags → those 6 features → 2 features

Both stages use:
  • Estimator : Logistic Regression with L1 regularization
  • CV        : 10-fold StratifiedKFold
  • Scoring   : accuracy (matches paper's reported metric)

The two important final features are:
  • Adverb_Adjective  (positive coefficient ≈ +2.73 → signals Fiction)
  • Adjective_Pronoun (negative coefficient ≈ −2.90 → signals Non-fiction)
"""

import warnings
# Suppress sklearn 1.8 deprecation of penalty= kwarg (liblinear still works
# correctly with penalty='l1'; the warning is about the kwarg name only).
warnings.filterwarnings('ignore', message='.*penalty.*deprecated.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*penalty.*deprecated.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Inconsistent values.*penalty.*', category=UserWarning)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Non-fiction categories in Brown Corpus (Table 1 of paper)
NF_CATEGORIES = {'news', 'reviews', 'hobbies', 'learned', 'government'}

# The 11 low-level feature names (used in "All Low-Level" feature set in Table 4)
LOW_LEVEL_FEATURES = [
    'Avg_Sentence_Length', 'Avg_Word_Length',
    'Std_Sentence_Length', 'Std_Word_Length',
    'Avg_Token_Type', 'Std_Token_Type',
    'Avg_Punct_Token_Type', 'Std_Punct_Token_Type',
    'Hyphen_Quote', 'Hyphen_Exclamation', 'Quote_Question',
]

# All 19 feature names
ALL_FEATURES = LOW_LEVEL_FEATURES + [
    'Adverb_Noun', 'Adverb_Pronoun', 'Adjective_Verb', 'Noun_Verb',
    'Verb_Pronoun', 'Adverb_Adjective', 'Adjective_Pronoun', 'Noun_Pronoun',
]

# The single list containing the features you want to analyze right now.
# Edit this list to eliminate or alter features for your custom analysis.
ACTIVE_FEATURES = [
    'Avg_Sentence_Length', 'Avg_Word_Length',
    'Std_Sentence_Length', 'Std_Word_Length',
    'Avg_Token_Type', 'Std_Token_Type',
    'Avg_Punct_Token_Type', 'Std_Punct_Token_Type',
    'Hyphen_Quote', 'Hyphen_Exclamation', 'Quote_Question',
    'Adverb_Noun', 'Adverb_Pronoun', 'Adjective_Verb', 'Noun_Verb',
    'Verb_Pronoun', 'Adverb_Adjective', 'Adjective_Pronoun', 'Noun_Pronoun',
]


def _make_target(df):
    """Map Category column → binary Target (0=Non-fiction, 1=Fiction)."""
    return (df['Category'].map(
        lambda c: 0 if c in NF_CATEGORIES else 1
    )).values


def _build_estimator():
    """Return the L1 Logistic Regression used in both RFECV stages.

    solver='liblinear' is the most reliable L1 solver for small datasets
    (sklearn recommends it for n_samples < ~10k with L1).
    """
    return LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        max_iter=1000,
        random_state=42,
    )


def _build_cv():
    """Return the 10-fold Stratified CV used in both RFECV stages."""
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def run_rfecv_stage(X_scaled, y, feature_names, stage_label):
    """Run one RFECV stage and print the results.

    Args:
        X_scaled     : np.ndarray — scaled feature matrix
        y            : np.ndarray — binary labels
        feature_names: list[str]  — feature names for X_scaled columns
        stage_label  : str        — label for printout

    Returns:
        selected_features : list[str] — names of selected features
        rfecv             : fitted RFECV object
    """
    rfecv = RFECV(
        estimator=_build_estimator(),
        step=1,
        cv=_build_cv(),
        scoring='accuracy',
        min_features_to_select=1,
        n_jobs=-1,
    )
    rfecv.fit(X_scaled, y)

    selected = np.array(feature_names)[rfecv.support_].tolist()
    cv_acc   = max(rfecv.cv_results_['mean_test_score'])

    print(f"\n{'='*60}")
    print(f"RFECV Stage: {stage_label}")
    print(f"{'='*60}")
    print(f"  Optimal # features : {rfecv.n_features_}")
    print(f"  Best CV accuracy   : {cv_acc:.4f}")
    print(f"  Selected features  : {selected}")

    return selected, rfecv


def run_two_stage_rfecv(df_brown_gold, df_brown_ptb):
    """Run the two-stage RFECV as described in the paper.

    Args:
        df_brown_gold : DataFrame with 19 features extracted using Brown gold tags.
        df_brown_ptb  : DataFrame with 19 features extracted using PTB auto tags.

    Returns:
        dict with keys:
          'six_features'  : list of 6 feature names selected in Stage 1
          'two_features'  : list of 2 feature names selected in Stage 2
          'coefficients'  : dict {feature_name: coefficient} from final L1 LR model
          'rfecv_stage1'  : fitted RFECV object from Stage 1
          'rfecv_stage2'  : fitted RFECV object from Stage 2
    """
    # ---------------------------------------------------------------
    # Stage 1: All 19 features on Brown GOLD tags → 6 features
    # ---------------------------------------------------------------
    y_gold = _make_target(df_brown_gold)
    X_gold = df_brown_gold[ALL_FEATURES].values
    scaler1 = StandardScaler()
    X_gold_scaled = scaler1.fit_transform(X_gold)

    six_features, rfecv1 = run_rfecv_stage(
        X_gold_scaled, y_gold, ALL_FEATURES,
        stage_label="Stage 1 — Brown Gold Tags (19 → 6 features)"
    )

    # ---------------------------------------------------------------
    # Stage 2: 6 selected features on Brown PTB auto tags → 2 features
    # ---------------------------------------------------------------
    y_ptb = _make_target(df_brown_ptb)
    X_ptb = df_brown_ptb[six_features].values
    scaler2 = StandardScaler()
    X_ptb_scaled = scaler2.fit_transform(X_ptb)

    two_features, rfecv2 = run_rfecv_stage(
        X_ptb_scaled, y_ptb, six_features,
        stage_label="Stage 2 — Brown PTB Auto Tags (6 → 2 features)"
    )

    # ---------------------------------------------------------------
    # Final L1 Logistic Regression on the 2 features (PTB tags)
    # — print the learned coefficients to verify signs match paper
    # ---------------------------------------------------------------
    X_final = df_brown_ptb[two_features].values
    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    lr_final = _build_estimator()
    lr_final.fit(X_final_scaled, y_ptb)

    coefs = dict(zip(two_features, lr_final.coef_[0]))
    print(f"\n  Final LR coefficients (positive → Fiction, negative → Non-fiction):")
    for feat, coef in coefs.items():
        direction = "Fiction" if coef > 0 else "Non-fiction"
        print(f"    {feat:25s}: {coef:+.4f}  → {direction}")

    return {
        'six_features':  six_features,
        'two_features':  two_features,
        'coefficients':  coefs,
        'rfecv_stage1':  rfecv1,
        'rfecv_stage2':  rfecv2,
        'scaler_final':  scaler_final,
        'lr_final':      lr_final,
    }
