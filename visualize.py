"""
visualize.py
============
Generates Figure 1, Figure 2, Figure 3, Figure 4 from:
  Qureshi et al., "A Simple Approach to Classify Fictional and Non-Fictional
  Genres", ACL Storytelling Workshop, 2019.

Figure 1 — Scatter plot: Adverb/Adjective vs Adjective/Pronoun
           coloured by genre (Non-fiction = circles, Fiction = crosses)

Figure 2 — Bar chart: average POS % (ADJ, ADV, NOUN, VERB, PRON)
           across fiction and non-fiction documents

Figure 3 — Bar chart: average Pronoun token % across documents

Figure 4 — Bar chart: average Punctuation token % across documents
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colour palette (one colour per subgenre) ─────────────────────────────────
SUBGENRE_COLOURS = {
    # Non-fiction (circles) — cool tones
    'news':         '#1f77b4',
    'reviews':      '#aec7e8',
    'hobbies':      '#17becf',
    'government':   '#2ca02c',
    'learned':      '#98df8a',
    # Fiction (crosses) — warm tones
    'fiction':      '#d62728',
    'mystery':      '#ff9896',
    'science_fiction': '#9467bd',
    'adventure':    '#ff7f0e',
    'romance':      '#ffbb78',
}

NF_SUBGENRES = {'news', 'reviews', 'hobbies', 'government', 'learned'}
F_SUBGENRES  = {'fiction', 'mystery', 'science_fiction', 'adventure', 'romance'}


# ── Figure 1 ─────────────────────────────────────────────────────────────────

def plot_figure1(df_brown, save_path='figure1_scatter.png'):
    """Scatter plot replicating Figure 1 of the paper.

    X-axis : Adverb / Adjective ratio
    Y-axis : Adjective / Pronoun ratio
    Markers: Non-fiction → 'o' (circle), Fiction → 'x' (cross)
    Colours: one per subgenre (10 subgenres total)

    Args:
        df_brown  : DataFrame with Brown Corpus features (PTB or Gold tags).
        save_path : output file path.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for subgenre, colour in SUBGENRE_COLOURS.items():
        sub = df_brown[df_brown['Category'] == subgenre]
        if sub.empty:
            continue
        marker = 'o' if subgenre in NF_SUBGENRES else 'x'
        label  = subgenre.replace('_', ' ').title()
        ax.scatter(
            sub['Adverb_Adjective'],
            sub['Adjective_Pronoun'],
            c=colour,
            marker=marker,
            s=60,
            linewidths=1.2,
            label=label,
            alpha=0.8,
        )

    # Legend: split into two columns (NF | F) to match paper layout
    nf_patches = [
        mpatches.Patch(color=SUBGENRE_COLOURS[s],
                       label=s.replace('_', ' ').title())
        for s in NF_SUBGENRES
    ]
    f_patches = [
        mpatches.Patch(color=SUBGENRE_COLOURS[s],
                       label=s.replace('_', ' ').title())
        for s in F_SUBGENRES
    ]
    legend1 = ax.legend(
        handles=nf_patches,
        title='Non-fiction  ○',
        loc='upper right',
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=f_patches,
        title='Fiction  ×',
        loc='upper left',
        fontsize=8,
        title_fontsize=8,
    )

    ax.set_xlabel('Adverb / Adjective ratio', fontsize=11)
    ax.set_ylabel('Adjective / Pronoun ratio', fontsize=11)
    ax.set_title('Figure 1 — Genre Cluster Plot\n(Adverb/Adjective vs Adjective/Pronoun)',
                 fontsize=12)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, None)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Figure 1] Saved to {save_path}")


# ── Figure 2 — POS distribution ──────────────────────────────────────────────

def plot_figure2(df_brown, save_path='figure2_pos.png'):
    """Bar chart: mean POS tag percentages for Fiction vs Non-fiction."""
    pos_cols = ['Adverb_Noun', 'Adverb_Pronoun', 'Adjective_Verb',
                'Noun_Verb', 'Verb_Pronoun', 'Adverb_Adjective',
                'Adjective_Pronoun', 'Noun_Pronoun']

    nf_mask = df_brown['Category'].isin(NF_SUBGENRES)
    means_nf = df_brown.loc[nf_mask, pos_cols].mean()
    means_f  = df_brown.loc[~nf_mask, pos_cols].mean()

    x = np.arange(len(pos_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, means_nf, width, label='Non-fiction', color='steelblue')
    ax.bar(x + width/2, means_f,  width, label='Fiction',     color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '/') for c in pos_cols], rotation=30, ha='right')
    ax.set_ylabel('Mean ratio value')
    ax.set_title('Figure 2 — POS Ratio Distributions: Fiction vs Non-fiction')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Figure 2] Saved to {save_path}")


# ── Figure 3 — Punctuation distribution ──────────────────────────────────────

def plot_figure3(df_brown, save_path='figure3_punct.png'):
    """Bar chart: mean punctuation ratios for Fiction vs Non-fiction."""
    punct_cols = ['Hyphen_Quote', 'Hyphen_Exclamation', 'Quote_Question',
                  'Avg_Punct_Token_Type', 'Std_Punct_Token_Type']

    nf_mask = df_brown['Category'].isin(NF_SUBGENRES)
    means_nf = df_brown.loc[nf_mask, punct_cols].mean()
    means_f  = df_brown.loc[~nf_mask, punct_cols].mean()

    x = np.arange(len(punct_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, means_nf, width, label='Non-fiction', color='steelblue')
    ax.bar(x + width/2, means_f,  width, label='Fiction',     color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in punct_cols], fontsize=9)
    ax.set_ylabel('Mean ratio value')
    ax.set_title('Figure 3 — Punctuation Ratio Distributions: Fiction vs Non-fiction')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Figure 3] Saved to {save_path}")


# ── Figure 4 — Low-level features ────────────────────────────────────────────

def plot_figure4(df_brown, save_path='figure4_lowlevel.png'):
    """Bar chart: mean low-level features for Fiction vs Non-fiction."""
    ll_cols = ['Avg_Sentence_Length', 'Avg_Word_Length',
               'Std_Sentence_Length', 'Std_Word_Length',
               'Avg_Token_Type', 'Std_Token_Type']

    nf_mask = df_brown['Category'].isin(NF_SUBGENRES)
    means_nf = df_brown.loc[nf_mask, ll_cols].mean()
    means_f  = df_brown.loc[~nf_mask, ll_cols].mean()

    x = np.arange(len(ll_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, means_nf, width, label='Non-fiction', color='steelblue')
    ax.bar(x + width/2, means_f,  width, label='Fiction',     color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in ll_cols], fontsize=9)
    ax.set_ylabel('Mean value')
    ax.set_title('Figure 4 — Low-Level Feature Distributions: Fiction vs Non-fiction')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Figure 4] Saved to {save_path}")

