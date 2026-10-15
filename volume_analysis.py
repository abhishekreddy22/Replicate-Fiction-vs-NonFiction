import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import brown

# Import from existing codebase
from data_handling.load_brown import (
    filtered_fileids, PUNCT_SET, HYPHEN_TOKENS, QUOTE_TOKENS, 
    EXCLAIM_TOKENS, QUESTION_TOKENS, _group_pos_counts, BROWN_TAG_GROUPS, PTB_TAG_GROUPS
)
from feature_selection import ALL_FEATURES, LOW_LEVEL_FEATURES
from experiments import _lr, _target_brown, _target_bnc, _metrics, build_feature_sets, _format_row, _print_table

# ── Feature extraction with sentence dropping ────────────────────────────────

def extract_reduced_document_features(fileid, tagset='brown', drop_ratio=0.5, seed=42):
    """Extract features after randomly dropping `drop_ratio` of sentences."""
    random.seed(seed)
    
    # Get original sentences
    all_sents = list(brown.sents(fileids=fileid))
    all_tagged_sents_gold = list(brown.tagged_sents(fileids=fileid)) if tagset == 'brown' else None
    
    num_sents = len(all_sents)
    keep_count = max(1, int(num_sents * (1.0 - drop_ratio)))
    keep_indices = sorted(random.sample(range(num_sents), k=keep_count))
    
    sents = [all_sents[i] for i in keep_indices]
    words = [w for sent in sents for w in sent]
    category = brown.categories(fileids=fileid)[0]

    # 1. Low-Level Normalized
    sent_lengths = [len(s) for s in sents]
    word_lengths = [len(w) for w in words if w.isalnum()]

    # 2. Low-Level Ratios
    chunk_size = 100
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    ttr_list, punct_ttr_list = [], []
    for chunk in chunks:
        if not chunk: continue
        ttr_list.append(len(chunk) / len(set(chunk)))
        punct_chunk = [w for w in chunk if w in PUNCT_SET]
        if punct_chunk:
            punct_ttr_list.append(len(punct_chunk) / len(set(punct_chunk)))

    from collections import Counter
    counts  = Counter(words)
    hyphens = sum(counts.get(t, 0) for t in HYPHEN_TOKENS)
    quotes  = sum(counts.get(t, 0) for t in QUOTE_TOKENS)
    exclaims   = sum(counts.get(t, 0) for t in EXCLAIM_TOKENS)
    questions  = sum(counts.get(t, 0) for t in QUESTION_TOKENS)

    # 3. High-Level POS Ratios
    if tagset == 'brown':
        tagged_sents = [all_tagged_sents_gold[i] for i in keep_indices]
        tagged_raw = [wt for sent in tagged_sents for wt in sent]
        raw_tags = [t.split('-')[0] if '-' in t else t for _, t in tagged_raw]
        pos = _group_pos_counts(raw_tags, BROWN_TAG_GROUPS)
    elif tagset == 'ptb':
        tagged_raw = nltk.pos_tag(words)
        raw_tags = [t for _, t in tagged_raw]
        pos = _group_pos_counts(raw_tags, PTB_TAG_GROUPS)
    else:
        raise ValueError(f"Unknown tagset: {tagset!r}")

    def ratio(num, den):
        return num / den if den > 0 else 0.0

    return {
        'FileID':               fileid,
        'Category':             category,
        'Avg_Sentence_Length':  np.mean(sent_lengths) if sent_lengths else 0.0,
        'Avg_Word_Length':      np.mean(word_lengths) if word_lengths else 0.0,
        'Std_Sentence_Length':  np.std(sent_lengths) if sent_lengths else 0.0,
        'Std_Word_Length':      np.std(word_lengths) if word_lengths else 0.0,
        'Avg_Token_Type':       np.mean(ttr_list) if ttr_list else 0.0,
        'Std_Token_Type':       np.std(ttr_list) if ttr_list else 0.0,
        'Avg_Punct_Token_Type': np.mean(punct_ttr_list) if punct_ttr_list else 0.0,
        'Std_Punct_Token_Type': np.std(punct_ttr_list) if punct_ttr_list else 0.0,
        'Hyphen_Quote':         ratio(hyphens, quotes),
        'Hyphen_Exclamation':   ratio(hyphens, exclaims),
        'Quote_Question':       ratio(quotes, questions),
        'Adverb_Noun':          ratio(pos['ADV'], pos['NOUN']),
        'Adverb_Pronoun':       ratio(pos['ADV'], pos['PRON']),
        'Adjective_Verb':       ratio(pos['ADJ'], pos['VERB']),
        'Noun_Verb':            ratio(pos['NOUN'], pos['VERB']),
        'Verb_Pronoun':         ratio(pos['VERB'], pos['PRON']),
        'Adverb_Adjective':     ratio(pos['ADV'], pos['ADJ']),
        'Adjective_Pronoun':    ratio(pos['ADJ'], pos['PRON']),
        'Noun_Pronoun':         ratio(pos['NOUN'], pos['PRON']),
    }

def preprocess_reduced_brown(tagset='brown', drop_ratio=0.5):
    print(f"Extracting features from Brown Corpus (dropping {drop_ratio*100}% sentences)...")
    data = [extract_reduced_document_features(f, tagset=tagset, drop_ratio=drop_ratio) for f in filtered_fileids]
    df = pd.DataFrame(data)
    return df

# ── Experiment 1: One-Class SVM on Fiction ─────────────────────────────────

def experiment_one_class_svm(df_brown_gold, df_bnc):
    print("\n" + "="*70)
    print("EXPERIMENT 1: One-Class SVM trained ONLY on Brown Fiction (Gold)")
    print("Testing on Brown + BNC to detect Non-Fiction as anomalies (-1)")
    print("="*70)
    
    # y = 1 for fiction, 0 for non-fiction
    y_brown = _target_brown(df_brown_gold)
    
    # Train only on fiction
    X_train = df_brown_gold[y_brown == 1][ALL_FEATURES].values
    
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    
    # One-Class SVM
    # nu controls the upper bound on the fraction of training errors (anomalies)
    # and lower bound of the fraction of support vectors.
    clf = OneClassSVM(gamma='scale', nu=0.05)
    clf.fit(X_train_s)
    
    # Test on full Brown
    X_brown_all_s = sc.transform(df_brown_gold[ALL_FEATURES].values)
    preds_brown = clf.predict(X_brown_all_s)
    
    # OneClassSVM outputs 1 for inlier (Fiction), -1 for outlier (Non-Fiction)
    # Convert true labels to match: 1 -> 1, 0 -> -1
    y_brown_true = np.where(y_brown == 1, 1, -1)
    
    print("\nResults on Brown Corpus (All 19 Features):")
    print(classification_report(y_brown_true, preds_brown, target_names=["Non-Fiction (-1)", "Fiction (1)"]))
    print(f"Accuracy: {accuracy_score(y_brown_true, preds_brown)*100:.2f}%")
    
    # Test on BNC (cross-corpus)
    y_bnc = _target_bnc(df_bnc)
    X_bnc_s = sc.transform(df_bnc[ALL_FEATURES].values)
    preds_bnc = clf.predict(X_bnc_s)
    y_bnc_true = np.where(y_bnc == 1, 1, -1)
    
    print("\nResults on BNC Corpus (All 19 Features):")
    print(classification_report(y_bnc_true, preds_bnc, target_names=["Non-Fiction (-1)", "Fiction (1)"]))
    print(f"Accuracy: {accuracy_score(y_bnc_true, preds_bnc)*100:.2f}%")

# ── Experiment 2: Reduced Data Volume Classification ────────────────────────

def experiment_reduced_volume(df_brown_reduced, six_features, two_features):
    print("\n" + "="*70)
    print("EXPERIMENT 2: 50% Reduced Document Volume in Brown Corpus")
    print("Setup (a) on Reduced Data (10 x 60/40 splits)")
    print("="*70)
    
    feature_sets = build_feature_sets(six_features, two_features)
    y = _target_brown(df_brown_reduced)
    rows = []
    
    from sklearn.model_selection import train_test_split
    
    for label, feats in feature_sets.items():
        X = df_brown_reduced[feats].values
        
        results = []
        majority_size = max(np.sum(y == 0), np.sum(y == 1))
        for seed in range(10):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            model = _lr()
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            results.append(_metrics(y_te, y_pred, majority_size, len(y)))
            
        keys = ['Accuracy', 'F1_NF', 'F1_F', 'Baseline', 'AccuracyGain']
        stats = {
            k: (np.mean([r[k] for r in results]), np.std([r[k] for r in results]))
            for k in keys
        }
        rows.append(_format_row(label, stats))
        
    _print_table(rows)

# ── Main Execution ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    from data_handling import preprocess_brown_corpus, preprocess_bnc_corpus
    from feature_selection import run_two_stage_rfecv
    
    BNC_ROOT = "Corpus/baby_bnc/Texts"
    
    print("Extracting Brown Gold...")
    df_brown_gold = preprocess_brown_corpus(tagset='brown')
    print("Extracting Brown PTB...")
    df_brown_ptb = preprocess_brown_corpus(tagset='ptb')
    print("Extracting BNC...")
    df_bnc = preprocess_bnc_corpus(BNC_ROOT)
    
    print("Running feature selection to get 6 and 2 feature sets...")
    rfecv_results = run_two_stage_rfecv(df_brown_gold, df_brown_ptb)
    best_6 = rfecv_results['six_features']
    best_2 = rfecv_results['two_features']
    
    experiment_one_class_svm(df_brown_gold, df_bnc)
    
    # Create the reduced dataset
    df_brown_reduced = preprocess_reduced_brown(tagset='brown', drop_ratio=0.5)
    
    experiment_reduced_volume(df_brown_reduced, best_6, best_2)
