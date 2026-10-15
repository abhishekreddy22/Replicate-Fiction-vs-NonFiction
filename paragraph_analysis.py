"""
paragraph_analysis.py
=====================
Investigates the impact of removing paragraphs on genre classification accuracy.
Evaluates three removal strategies:
  1. Random paragraphs removed
  2. First few paragraphs removed
  3. Last few paragraphs removed
Across drop ratios: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% (plus 0% baseline).

Evaluates on Brown Corpus + BNC Baby (55 docs) using 10 random 60/40 splits (Setup B).
Tests all four feature sets:
  - All 19 Features
  - All Low-Level (11 Features)
  - 6 Selected Features
  - 2 Selected Features
"""

import os
import random
import pickle
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_handling.load_brown import (
    filtered_fileids, PUNCT_SET, HYPHEN_TOKENS, QUOTE_TOKENS,
    EXCLAIM_TOKENS, QUESTION_TOKENS, PTB_TAG_GROUPS, _group_pos_counts
)
from feature_selection import LOW_LEVEL_FEATURES, ALL_FEATURES, NF_CATEGORIES

BNC_ROOT = "Corpus/baby_bnc/Texts"
CACHE_FILE = "scratch/tagged_paragraphs_cache.pkl"

# Six and two selected features
SIX_FEATURES = [
    'Avg_Sentence_Length', 'Std_Sentence_Length', 'Std_Word_Length',
    'Std_Token_Type', 'Hyphen_Quote', 'Adverb_Noun'
]
TWO_FEATURES = ['Adverb_Adjective', 'Adjective_Pronoun']

FEATURE_SETS = {
    'All 19 Features': ALL_FEATURES,
    'All Low-Level (11)': LOW_LEVEL_FEATURES,
    '6 Features': SIX_FEATURES,
    '2 Features': TWO_FEATURES,
}


def _local_name(tag):
    return tag.rsplit('}', 1)[-1]


def _read_bnc_paragraphs(xml_file):
    """Parse BNC XML file into list of paragraphs, each containing sentences of tokens."""
    paras = []
    current_para = []
    current_sent = []

    for event, element in ET.iterparse(xml_file, events=('start', 'end')):
        tag = _local_name(element.tag)
        if event == 'end' and tag in {'w', 'c'}:
            w = (element.text or '').strip()
            if w:
                current_sent.append(w)
            element.clear()
        elif event == 'end' and tag == 's':
            if current_sent:
                current_para.append(current_sent)
                current_sent = []
            element.clear()
        elif event == 'end' and tag == 'p':
            if current_sent:
                current_para.append(current_sent)
                current_sent = []
            if current_para:
                paras.append(current_para)
                current_para = []
            element.clear()

    if current_sent:
        current_para.append(current_sent)
    if current_para:
        paras.append(current_para)

    # Fallback if no <p> was found
    if not paras:
        all_words, all_sents = [], []
        # fallback simple
        paras = [current_para] if current_para else []
    return paras


def load_and_tag_corpus():
    """Load and tag paragraphs for all 324 Brown docs and 55 BNC docs."""
    os.makedirs('scratch', exist_ok=True)
    if os.path.exists(CACHE_FILE):
        print(f"[Cache] Loading pre-tagged paragraphs from {CACHE_FILE} ...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print("[Corpus] Pre-tagging Brown Corpus paragraphs (324 docs) ...")
    docs = []

    for fid in filtered_fileids:
        cat = brown.categories(fileids=fid)[0]
        # Target: 0 for NF, 1 for Fiction
        target = 0 if cat in NF_CATEGORIES else 1
        raw_paras = brown.paras(fileids=fid)
        
        tagged_paras = []
        for p in raw_paras:
            tagged_p = []
            for sent in p:
                if sent:
                    tagged_p.append(nltk.pos_tag(sent))
            if tagged_p:
                tagged_paras.append(tagged_p)
                
        docs.append({
            'doc_id': fid,
            'source': 'brown',
            'category': cat,
            'target': target,
            'paras': tagged_paras
        })

    print("[Corpus] Pre-tagging BNC Baby paragraphs (55 docs) ...")
    bnc_subdirs = [('fic', 1), ('aca', 0)]
    for sub, target in bnc_subdirs:
        xml_dir = Path(BNC_ROOT) / sub
        for xml_file in sorted(xml_dir.glob('*.xml')):
            raw_paras = _read_bnc_paragraphs(xml_file)
            tagged_paras = []
            for p in raw_paras:
                tagged_p = []
                for sent in p:
                    if sent:
                        tagged_p.append(nltk.pos_tag(sent))
                if tagged_p:
                    tagged_paras.append(tagged_p)
            docs.append({
                'doc_id': xml_file.stem,
                'source': 'bnc',
                'category': sub,
                'target': target,
                'paras': tagged_paras
            })

    print(f"[Cache] Saving {len(docs)} pre-tagged documents to {CACHE_FILE} ...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(docs, f)

    return docs


def extract_features_from_paras(tagged_paras):
    """Compute the 19 stylometric features from a subset of tagged paragraphs."""
    sents = [sent for p in tagged_paras for sent in p]
    if not sents:
        # Emergency fallback for empty document
        return {f: 0.0 for f in ALL_FEATURES}

    words = [w for sent in sents for w, _ in sent]
    tags = [t for sent in sents for _, t in sent]

    sent_lengths = [len(s) for s in sents]
    word_lengths = [len(w) for w in words if w.isalnum()]

    chunk_size = 100
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    ttr_list, punct_ttr_list = [], []
    for chunk in chunks:
        if not chunk:
            continue
        ttr_list.append(len(chunk) / len(set(chunk)))
        punct_chunk = [w for w in chunk if w in PUNCT_SET]
        if punct_chunk:
            punct_ttr_list.append(len(punct_chunk) / len(set(punct_chunk)))

    counts = Counter(words)
    hyphens = sum(counts.get(t, 0) for t in HYPHEN_TOKENS)
    quotes = sum(counts.get(t, 0) for t in QUOTE_TOKENS)
    exclaims = sum(counts.get(t, 0) for t in EXCLAIM_TOKENS)
    questions = sum(counts.get(t, 0) for t in QUESTION_TOKENS)

    pos = _group_pos_counts(tags, PTB_TAG_GROUPS)

    def ratio(num, den):
        return num / den if den > 0 else 0.0

    return {
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


def select_paragraphs(paras, strategy, drop_ratio, seed=42):
    """Select paragraphs to KEEP after dropping drop_ratio."""
    n = len(paras)
    if n <= 1 or drop_ratio <= 0.0:
        return paras

    # Drop k paragraphs, keeping at least 1
    k = max(1, int(round(n * drop_ratio)))
    k = min(k, n - 1)

    if strategy == 'first':
        # Remove the first k paragraphs
        return paras[k:]
    elif strategy == 'last':
        # Remove the last k paragraphs
        return paras[:n - k]
    elif strategy == 'random':
        # Remove k random paragraphs
        rng = random.Random(seed)
        remove_indices = set(rng.sample(range(n), k))
        return [paras[i] for i in range(n) if i not in remove_indices]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_dataset(df, y, n_splits=10, test_size=0.4):
    """Run 10 random 60/40 splits and compute accuracy for each feature set."""
    scores = {}
    for feat_name, feats in FEATURE_SETS.items():
        X = df[feats].values
        split_accs = []
        for seed in range(n_splits):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=seed
            )
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            model = LogisticRegression(
                penalty='l1', solver='liblinear', C=1.0, max_iter=1000, random_state=42
            )
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)
            split_accs.append(accuracy_score(y_te, preds) * 100)

        scores[feat_name] = {
            'mean': np.mean(split_accs),
            'std': np.std(split_accs)
        }
    return scores


def run_experiment():
    docs = load_and_tag_corpus()
    y = np.array([d['target'] for d in docs])

    drop_percentages = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    strategies = ['random', 'first', 'last']

    records = []

    print("\n" + "="*70)
    print("RUNNING PARAGRAPH REMOVAL EXPERIMENT")
    print("Corpus: Brown + BNC Baby (379 documents)")
    print("Setup: 10 repeated 60/40 stratified train/test splits")
    print("="*70)

    for strat in strategies:
        print(f"\nStrategy: {strat.upper()}")
        for pct in drop_percentages:
            feat_rows = []
            for i, doc in enumerate(docs):
                kept_paras = select_paragraphs(
                    doc['paras'], strategy=strat, drop_ratio=pct, seed=42 + i
                )
                feat_dict = extract_features_from_paras(kept_paras)
                feat_rows.append(feat_dict)

            df_features = pd.DataFrame(feat_rows)
            scores = evaluate_dataset(df_features, y)

            for feat_name, score_data in scores.items():
                records.append({
                    'strategy': strat,
                    'drop_percentage': int(round(pct * 100)),
                    'feature_set': feat_name,
                    'mean_accuracy': score_data['mean'],
                    'std_accuracy': score_data['std']
                })
            
            acc_19 = scores['All 19 Features']['mean']
            acc_2 = scores['2 Features']['mean']
            print(f"  Drop {int(round(pct*100)):2d}% | All 19: {acc_19:5.2f}% | 2 Feats: {acc_2:5.2f}%")

    res_df = pd.DataFrame(records)
    res_df.to_csv('scratch/paragraph_removal_results.csv', index=False)
    print("\n[Done] Saved numerical results to scratch/paragraph_removal_results.csv")

    return res_df


def plot_results(res_df):
    """Generate high-quality plots for the experiment."""
    os.makedirs('plots', exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Color palette
    colors = {
        'random': '#1f77b4',   # Blue
        'first':  '#ff7f0e',   # Orange
        'last':   '#2ca02c',   # Green
    }
    markers = {
        'random': 'o',
        'first':  's',
        'last':   '^',
    }
    strat_labels = {
        'random': 'Random Paragraphs Removed',
        'first':  'First Few Paragraphs Removed',
        'last':   'Last Few Paragraphs Removed',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    feature_sets = list(FEATURE_SETS.keys())

    for idx, f_name in enumerate(feature_sets):
        ax = axes[idx]
        sub = res_df[res_df['feature_set'] == f_name]

        # Baseline accuracy at 0% drop
        baseline_acc = sub[sub['drop_percentage'] == 0]['mean_accuracy'].mean()
        ax.axhline(baseline_acc, color='gray', linestyle='--', alpha=0.7,
                   label=f'0% Drop Baseline ({baseline_acc:.1f}%)')

        for strat in ['random', 'first', 'last']:
            strat_data = sub[sub['strategy'] == strat].sort_values('drop_percentage')
            x = strat_data['drop_percentage'].values
            y = strat_data['mean_accuracy'].values
            err = strat_data['std_accuracy'].values

            ax.plot(x, y, color=colors[strat], marker=markers[strat],
                    linewidth=2, markersize=6, label=strat_labels[strat])
            ax.fill_between(x, y - err, y + err, color=colors[strat], alpha=0.15)

        ax.set_title(f"Feature Set: {f_name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_xlabel("Paragraphs Removed (%)", fontsize=11)
        ax.set_xticks(range(0, 100, 10))
        ax.set_ylim(80, 100)
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=9, frameon=True)

    fig.suptitle("Effect of Paragraph Removal on Genre Classification Accuracy (Brown + BNC Baby)",
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = 'plots/paragraph_removal_accuracy.png'
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved main 4-panel figure to {plot_path}")

    # Also generate a consolidated summary comparison for All 19 Features
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    sub19 = res_df[res_df['feature_set'] == 'All 19 Features']
    base19 = sub19[sub19['drop_percentage'] == 0]['mean_accuracy'].mean()
    ax2.axhline(base19, color='black', linestyle='--', linewidth=1.5,
                label=f'Full Document Baseline ({base19:.2f}%)')

    for strat in ['random', 'first', 'last']:
        s_data = sub19[sub19['strategy'] == strat].sort_values('drop_percentage')
        x = s_data['drop_percentage'].values
        y = s_data['mean_accuracy'].values
        err = s_data['std_accuracy'].values
        ax2.errorbar(x, y, yerr=err, color=colors[strat], marker=markers[strat],
                     capsize=4, linewidth=2, markersize=7, label=strat_labels[strat])

    ax2.set_title("Paragraph Removal Effect on All 19 Stylometric Features", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Percentage of Paragraphs Removed from Each Document (%)", fontsize=11)
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax2.set_xticks(range(0, 100, 10))
    ax2.set_ylim(85, 100)
    ax2.legend(loc='lower left', fontsize=10, frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.7)

    fig2.tight_layout()
    summary_path = 'plots/paragraph_removal_summary.png'
    fig2.savefig(summary_path, dpi=200)
    plt.close(fig2)
    print(f"[Plot] Saved summary plot to {summary_path}")


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)
