"""
load_brown.py
=============
Feature extraction from the Brown Corpus (NLTK).

Implements the exact 19 stylometric features described in:
  Qureshi et al., "A Simple Approach to Classify Fictional and Non-Fictional Genres"
  ACL Storytelling Workshop, 2019.

Features (Table 2 in paper):
  Low-Level Normalized  (4): Avg/Std Sentence Length, Avg/Std Word Length
  Low-Level Ratios      (7): Avg/Std Token-Type Ratio, Avg/Std Punct Token-Type,
                             Hyphen/Quote, Hyphen/Exclamation, Quote/Question
  High-Level POS Ratios (8): Adv/Noun, Adv/Pron, Adj/Verb, Noun/Verb,
                             Verb/Pron, Adv/Adj, Adj/Pron, Noun/Pron

Tagsets:
  'brown' — Gold-standard Brown Corpus tags (87 tags, suffix-stripped).
  'ptb'   — Automatic Penn Treebank tags from NLTK averaged perceptron tagger.
"""

import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import brown

nltk.download('brown', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ---------------------------------------------------------------------------
# Corpus filter — paper uses 10 subgenres, excluding 5 literary/editorial
# categories (humor, religion, belles_lettres, lore, editorial).
# Result: 324 documents (207 non-fiction, 117 fiction).
# ---------------------------------------------------------------------------
EXCLUDED_CATEGORIES = {'humor', 'religion', 'belles_lettres', 'lore', 'editorial'}

filtered_fileids = [
    f for f in brown.fileids()
    if not any(cat in EXCLUDED_CATEGORIES for cat in brown.categories(f))
]

# ---------------------------------------------------------------------------
# POS tag groupings (Table 3 of paper)
# ---------------------------------------------------------------------------

# Brown gold-standard tags (suffix-stripped, e.g. JJ-HL → JJ)
BROWN_TAG_GROUPS = {
    'ADJ':  {'JJ', 'JJR', 'JJS', 'JJT'},
    'ADV':  {'RB', 'RBR', 'WRB', 'RBT', 'RN', 'RP', 'NR'},
    'NOUN': {'NN', 'NNS', 'NN$', 'NNS$', 'NP', 'NP$', 'NPS', 'NPS$'},
    'VERB': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'PRON': {'PN', 'PN$', 'PP$', 'PPL', 'PPLS', 'PPO',
             'PPS', 'PPSS', 'PRP', 'PRP$', 'WP$', 'WPO', 'WPS'},
}

# Penn Treebank tags (NLTK averaged perceptron tagger output)
PTB_TAG_GROUPS = {
    'ADJ':  {'JJ', 'JJR', 'JJS'},
    'ADV':  {'RB', 'RBR', 'RBS', 'WRB'},
    'NOUN': {'NN', 'NNS', 'NNP', 'NNPS'},
    'VERB': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'PRON': {'PRP', 'WP'},
}

# ---------------------------------------------------------------------------
# Punctuation set — from Figure 4 of the paper (exact list used for counting)
# ` - , ; : ! ? / . ' ( ) [ ] { } $ * & % +
# Plus Brown-specific double-tokens: -- `` ''
# ---------------------------------------------------------------------------
PUNCT_SET = {
    '`', '-', ',', ';', ':', '!', '?', '/', '.', "'",
    '(', ')', '[', ']', '{', '}', '$', '*', '&', '%', '+',
    '--', '``', "''",
    # Unicode equivalents (same set as load_bnc.py for consistency)
    '\u2014',   # em-dash  —
    '\u2013',   # en-dash  –
    '\u201c',   # left double quote  "
    '\u201d',   # right double quote "
    '\u2018',   # left single quote  '
    '\u2019',   # right single quote '
}

# Hyphens: single - and double --
HYPHEN_TOKENS  = {'-', '--'}
# Quotes:  straight ' and " and Brown ligatures `` ''
QUOTE_TOKENS   = {"'", '"', '``', "''"}
EXCLAIM_TOKENS = {'!'}
QUESTION_TOKENS = {'?'}


def _group_pos_counts(tag_list, tag_groups):
    """Count tags by broad POS category using the given grouping."""
    grouped = {cat: 0 for cat in tag_groups}
    for tag in tag_list:
        for cat, tags in tag_groups.items():
            if tag in tags:
                grouped[cat] += 1
                break
    return grouped


def extract_document_features(fileid, tagset='brown'):
    """Extract the 19 stylometric features from a single Brown Corpus document.

    Args:
        fileid:  Brown corpus file identifier (e.g. 'ca01').
        tagset:  'brown' uses gold-standard annotations;
                 'ptb'   uses NLTK averaged perceptron tagger (PTB tagset).

    Returns:
        dict with 19 feature values plus 'FileID' and 'Category'.
    """
    words    = list(brown.words(fileids=fileid))
    sents    = list(brown.sents(fileids=fileid))
    category = brown.categories(fileids=fileid)[0]

    # ------------------------------------------------------------------
    # 1. Low-Level Normalized features
    # ------------------------------------------------------------------
    sent_lengths = [len(s) for s in sents]
    # Word length: only alphanumeric tokens (no punctuation)
    word_lengths = [len(w) for w in words if w.isalnum()]

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
        # Token/Type ratio = tokens / unique tokens
        ttr_list.append(len(chunk) / len(set(chunk)))

        # Punctuation Token/Type ratio within the same chunk
        punct_chunk = [w for w in chunk if w in PUNCT_SET]
        if punct_chunk:
            punct_ttr_list.append(len(punct_chunk) / len(set(punct_chunk)))

    counts  = Counter(words)
    hyphens = sum(counts.get(t, 0) for t in HYPHEN_TOKENS)
    quotes  = sum(counts.get(t, 0) for t in QUOTE_TOKENS)
    exclaims   = sum(counts.get(t, 0) for t in EXCLAIM_TOKENS)
    questions  = sum(counts.get(t, 0) for t in QUESTION_TOKENS)

    # ------------------------------------------------------------------
    # 3. High-Level POS Ratio features
    # ------------------------------------------------------------------
    if tagset == 'brown':
        tagged_raw = brown.tagged_words(fileids=fileid)
        # Strip suffixes like -HL, -TL, -NC (only strip first hyphen segment)
        raw_tags = [t.split('-')[0] if '-' in t else t for _, t in tagged_raw]
        pos = _group_pos_counts(raw_tags, BROWN_TAG_GROUPS)
    elif tagset == 'ptb':
        tagged_raw = nltk.pos_tag(words)
        raw_tags = [t for _, t in tagged_raw]
        pos = _group_pos_counts(raw_tags, PTB_TAG_GROUPS)
    else:
        raise ValueError(f"Unknown tagset: {tagset!r}. Use 'brown' or 'ptb'.")

    def ratio(num, den):
        return num / den if den > 0 else 0.0

    return {
        'FileID':               fileid,
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


def preprocess_brown_corpus(tagset='brown'):
    """Preprocess the Brown Corpus and return a 19-feature DataFrame.

    Args:
        tagset: 'brown' for gold-standard tags, 'ptb' for NLTK automatic tags.

    Returns:
        pd.DataFrame with columns [FileID, Category, <19 features>].
    """
    data = [extract_document_features(f, tagset=tagset) for f in filtered_fileids]
    df = pd.DataFrame(data)
    print(f"[Brown/{tagset}] {len(df)} documents, categories: "
          f"{df['Category'].value_counts().to_dict()}")
    return df
