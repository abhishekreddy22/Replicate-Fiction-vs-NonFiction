"""
load_bnc.py
===========
Feature extraction from the BNC Baby corpus (XML format).

Implements the exact 19 stylometric features described in:
  Qureshi et al., "A Simple Approach to Classify Fictional and Non-Fictional Genres"
  ACL Storytelling Workshop, 2019.

BNC Baby partitioning (Section 3 of paper):
  Training/testing : fic (25 docs) + aca (30 docs)  = 55 documents
  External eval    : news (97 docs) — Table 5 of paper
  Excluded         : dem (30 spoken) — not used in paper

Tagging:
  The paper used a Stanford POS Tagger to produce Penn Treebank tags.
  Here we use NLTK's averaged perceptron tagger, which is the closest
  freely available equivalent (same 36-tag PTB tagset, minor differences
  on boundary cases).

  The Brown bigram/unigram N-gram tagger is NOT used for BNC — the paper
  never used it for BNC. Only PTB automatic tags are used.
"""

import pandas as pd
import numpy as np
import nltk
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ---------------------------------------------------------------------------
# POS tag groupings — Penn Treebank (Table 3 of paper)
# ---------------------------------------------------------------------------
PTB_TAG_GROUPS = {
    'ADJ':  {'JJ', 'JJR', 'JJS'},
    'ADV':  {'RB', 'RBR', 'RBS', 'WRB'},
    'NOUN': {'NN', 'NNS', 'NNP', 'NNPS'},
    'VERB': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'PRON': {'PRP', 'WP'},
}

# ---------------------------------------------------------------------------
# Punctuation set — from Figure 4 of the paper
# ` - , ; : ! ? / . ' ( ) [ ] { } $ * & % +
# Plus unicode variants common in BNC XML
# ---------------------------------------------------------------------------
PUNCT_SET = {
    '`', '-', ',', ';', ':', '!', '?', '/', '.', "'",
    '(', ')', '[', ']', '{', '}', '$', '*', '&', '%', '+',
    '--', '``', "''",
    # Unicode equivalents present in BNC XML
    '\u2014',   # em-dash  —
    '\u2013',   # en-dash  –
    '\u201c',   # left double quote  "
    '\u201d',   # right double quote "
    '\u2018',   # left single quote  '
    '\u2019',   # right single quote '
}

HYPHEN_TOKENS   = {'-', '--', '\u2014', '\u2013'}          # - -- — –
QUOTE_TOKENS    = {"'", '"', '``', "''", '\u201c', '\u201d', '\u2018', '\u2019'}
EXCLAIM_TOKENS  = {'!'}
QUESTION_TOKENS = {'?'}


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _local_name(tag):
    """Strip XML namespace prefix from a tag string."""
    return tag.rsplit('}', 1)[-1]


def _read_bnc_document(xml_file):
    """Parse a BNC Baby XML file and return (words, sentences).

    BNC XML uses <w> for word tokens and <c> for punctuation tokens,
    grouped into <s> (sentence) elements.

    Returns:
        words : list of str   — all tokens in document order
        sents : list[list[str]] — tokens grouped by sentence boundary
    """
    words, sents, current_sentence = [], [], []

    for event, element in ET.iterparse(xml_file, events=('start', 'end')):
        tag = _local_name(element.tag)

        if event == 'end' and tag in {'w', 'c'}:
            # BNC tokens can have leading/trailing whitespace in XML text
            word = (element.text or '').strip()
            if word:
                words.append(word)
                current_sentence.append(word)
            element.clear()

        elif event == 'end' and tag == 's':
            if current_sentence:
                sents.append(current_sentence)
                current_sentence = []
            element.clear()

    # Flush any sentence not closed by an </s> tag
    if current_sentence:
        sents.append(current_sentence)

    # Fallback: treat entire document as one sentence
    if not sents and words:
        sents = [words]

    return words, sents


# ---------------------------------------------------------------------------
# POS grouping helper
# ---------------------------------------------------------------------------

def _group_pos_counts(tag_list, tag_groups):
    """Count tags by broad POS category using the given grouping."""
    grouped = {cat: 0 for cat in tag_groups}
    for tag in tag_list:
        for cat, tags in tag_groups.items():
            if tag in tags:
                grouped[cat] += 1
                break
    return grouped


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_document_features(xml_file, category):
    """Extract the 19 stylometric features from one BNC Baby XML document.

    Tagging uses NLTK averaged perceptron tagger (PTB tagset), matching
    the paper's use of a PTB-based automatic tagger on BNC data.

    Args:
        xml_file : path-like — path to BNC XML file
        category : str       — BNC category label ('fic', 'aca', 'news')

    Returns:
        dict of 19 feature values + 'FileID' and 'Category',
        or None if the document contains no tokens.
    """
    words, sents = _read_bnc_document(xml_file)
    if not words:
        return None

    # ------------------------------------------------------------------
    # 1. Low-Level Normalized features
    # ------------------------------------------------------------------
    sent_lengths = [len(s) for s in sents]
    word_lengths  = [len(w) for w in words if w.isalnum()]

    # ------------------------------------------------------------------
    # 2. Low-Level Ratio features
    #    100-word non-overlapping chunks (Kessler et al., 1997)
    # ------------------------------------------------------------------
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

    counts    = Counter(words)
    hyphens   = sum(counts.get(t, 0) for t in HYPHEN_TOKENS)
    quotes    = sum(counts.get(t, 0) for t in QUOTE_TOKENS)
    exclaims  = sum(counts.get(t, 0) for t in EXCLAIM_TOKENS)
    questions = sum(counts.get(t, 0) for t in QUESTION_TOKENS)

    # ------------------------------------------------------------------
    # 3. High-Level POS Ratio features — PTB tags only
    # ------------------------------------------------------------------
    tagged_raw = nltk.pos_tag(words)
    raw_tags   = [t for _, t in tagged_raw]
    pos = _group_pos_counts(raw_tags, PTB_TAG_GROUPS)

    def ratio(num, den):
        return num / den if den > 0 else 0.0

    return {
        'FileID':               Path(xml_file).stem,
        'Category':             category,
        # Low-Level Normalized
        'Avg_Sentence_Length':  np.mean(sent_lengths),
        'Avg_Word_Length':      np.mean(word_lengths) if word_lengths else 0.0,
        'Std_Sentence_Length':  np.std(sent_lengths),
        'Std_Word_Length':      np.std(word_lengths) if word_lengths else 0.0,
        # Low-Level Ratios (TTR)
        'Avg_Token_Type':       np.mean(ttr_list) if ttr_list else 0.0,
        'Std_Token_Type':       np.std(ttr_list) if ttr_list else 0.0,
        'Avg_Punct_Token_Type': np.mean(punct_ttr_list) if punct_ttr_list else 0.0,
        'Std_Punct_Token_Type': np.std(punct_ttr_list) if punct_ttr_list else 0.0,
        # Low-Level Ratios (Punctuation)
        'Hyphen_Quote':         ratio(hyphens, quotes),
        'Hyphen_Exclamation':   ratio(hyphens, exclaims),
        'Quote_Question':       ratio(quotes, questions),
        # High-Level POS Ratios
        'Adverb_Noun':          ratio(pos['ADV'], pos['NOUN']),
        'Adverb_Pronoun':       ratio(pos['ADV'], pos['PRON']),
        'Adjective_Verb':       ratio(pos['ADJ'], pos['VERB']),
        'Noun_Verb':            ratio(pos['NOUN'], pos['VERB']),
        'Verb_Pronoun':         ratio(pos['VERB'], pos['PRON']),
        'Adverb_Adjective':     ratio(pos['ADV'], pos['ADJ']),
        'Adjective_Pronoun':    ratio(pos['ADJ'], pos['PRON']),
        'Noun_Pronoun':         ratio(pos['NOUN'], pos['PRON']),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_bnc_corpus(bnc_root):
    """Load BNC Baby fic + aca documents and return a 19-feature DataFrame.

    This is the training/testing split used in Table 4 of the paper:
      fic  : 25 documents (Fiction     → label 1)
      aca  : 30 documents (Non-fiction → label 0)
    Total  : 55 documents

    Spoken documents (dem) are excluded (not in paper).
    News documents (news) are reserved for Table 5 — use load_bnc_news().

    Args:
        bnc_root : str or Path — directory containing 'fic', 'aca', 'news', 'dem' subdirs.

    Returns:
        pd.DataFrame with columns [FileID, Category, <19 features>].
    """
    bnc_root = Path(bnc_root)
    data = []

    for category in ['fic', 'aca']:
        cat_dir = bnc_root / category
        if not cat_dir.exists():
            raise FileNotFoundError(
                f"Expected '{category}' directory not found under {bnc_root}"
            )
        for xml_file in sorted(cat_dir.rglob('*.xml')):
            feat = extract_document_features(xml_file, category)
            if feat is not None:
                data.append(feat)

    if not data:
        raise FileNotFoundError(
            f"No BNC Baby XML files found under {bnc_root}; "
            "expected 'fic' and 'aca' subdirectories."
        )

    df = pd.DataFrame(data)
    print(f"[BNC/ptb] {len(df)} documents, categories: "
          f"{df['Category'].value_counts().to_dict()}")
    return df


def load_bnc_news(bnc_root):
    """Load BNC Baby news documents for external (Table 5) evaluation.

    Returns 97 news documents, all unlabelled (no Target column).
    These are tagged with the same PTB tagger used for fic/aca.

    Args:
        bnc_root : str or Path — directory containing 'news' subdir.

    Returns:
        pd.DataFrame with columns [FileID, Category, <19 features>].
    """
    bnc_root = Path(bnc_root)
    news_dir = bnc_root / 'news'
    if not news_dir.exists():
        raise FileNotFoundError(f"'news' directory not found under {bnc_root}")

    data = []
    for xml_file in sorted(news_dir.rglob('*.xml')):
        feat = extract_document_features(xml_file, 'news')
        if feat is not None:
            data.append(feat)

    df = pd.DataFrame(data)
    print(f"[BNC/news/ptb] {len(df)} documents")
    return df