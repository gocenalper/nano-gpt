#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nano-gpt technical documentation PDF generator.

This script produces a multi-page, polished PDF that visually explains the
character-level GPT (Transformer) implementation inside gpt_scratch.ipynb.
Everything is drawn with matplotlib (no external PDF tooling required).

Run:
    python docs/generate_pdf.py
Output:
    docs/nano-gpt-documentation.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Theme / color palette  (modern dark navy + vivid accents)
# ---------------------------------------------------------------------------
BG        = "#0d1117"   # background (GitHub dark)
PANEL     = "#161b22"   # panel
INK       = "#e6edf3"   # primary text
MUTED     = "#8b949e"   # secondary text
ACCENT    = "#58a6ff"   # blue accent
ACCENT2   = "#bc8cff"   # purple accent
GREEN     = "#3fb950"   # green
ORANGE    = "#f0883e"   # orange
PINK      = "#f778ba"   # pink
YELLOW    = "#e3b341"   # yellow
GRID      = "#30363d"   # thin line

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
})

PAGE_W, PAGE_H = 8.27, 11.69   # A4 (inch)


def new_page():
    """Return a full-page, axis-free canvas (0-100 coordinate system)."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(BG)
    return fig, ax


def box(ax, x, y, w, h, text="", fc=PANEL, ec=GRID, tc=INK, fs=10,
        weight="normal", lw=1.4, rounding=0.025, ha="center", va="center",
        align="center", family="DejaVu Sans"):
    """Rounded-corner box + centered text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding*100}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
        mutation_aspect=1, zorder=2,
    )
    ax.add_patch(patch)
    if text:
        tx = x + w / 2 if align == "center" else x + 1.5
        ax.text(tx, y + h / 2, text, ha=ha, va=va, fontsize=fs,
                color=tc, weight=weight, zorder=3, wrap=True, family=family)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color=ACCENT, lw=2.0, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, zorder=1,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls,
    )
    ax.add_patch(a)


def pill(ax, x, y, text, fc=ACCENT, tc=BG, fs=8.5, w=None):
    w = w if w else 2.0 + len(text) * 0.72
    box(ax, x, y, w, 3.0, text, fc=fc, ec=fc, tc=tc, fs=fs, weight="bold",
        rounding=0.012)
    return x + w


def header(ax, kicker, title, num):
    """Page title band."""
    ax.add_patch(Rectangle((0, 92), 100, 8, color=PANEL, zorder=0))
    ax.add_patch(Rectangle((0, 92), 1.4, 8, color=ACCENT, zorder=1))
    ax.text(5, 96.6, kicker, fontsize=9, color=ACCENT, weight="bold")
    ax.text(5, 93.6, title, fontsize=16, color=INK, weight="bold")
    ax.text(95, 95, f"{num:02d}", fontsize=22, color=GRID, weight="bold",
            ha="right", va="center")


def footer(ax, page):
    ax.add_line(Line2D([5, 95], [4, 4], color=GRID, lw=1))
    ax.text(5, 2.4, "nano-gpt  ·  Character-Level GPT", fontsize=8,
            color=MUTED)
    ax.text(95, 2.4, f"page {page}", fontsize=8, color=MUTED, ha="right")


def bullet(ax, x, y, title, body, color=ACCENT, tw=88):
    ax.add_patch(Circle((x, y + 0.35), 0.55, color=color, zorder=3))
    ax.text(x + 2, y + 0.7, title, fontsize=10.5, color=INK, weight="bold",
            va="center")
    if body:
        ax.text(x + 2, y - 1.4, body, fontsize=9, color=MUTED, va="center",
                wrap=True)


# ===========================================================================
# PAGE 1 — COVER
# ===========================================================================
def page_cover(pdf):
    fig, ax = new_page()

    # decorative background blobs
    ax.add_patch(Rectangle((0, 0), 100, 100, color=BG, zorder=0))
    for i, c in enumerate([ACCENT, ACCENT2, PINK]):
        ax.add_patch(Circle((78 + i * 4, 80 - i * 3), 16 - i * 3,
                     color=c, alpha=0.08, zorder=0))

    # top label
    ax.text(10, 86, "T E C H N I C A L   D O C U M E N T A T I O N", fontsize=11,
            color=ACCENT, weight="bold")
    ax.add_line(Line2D([10, 47], [84.5, 84.5], color=ACCENT, lw=2))

    # title
    ax.text(10, 74, "nano-gpt", fontsize=58, color=INK, weight="bold")
    ax.text(10, 66, "A Character-Level GPT\n(Transformer) From Scratch", fontsize=22,
            color=MUTED, weight="bold", linespacing=1.2, va="top")

    # description
    ax.text(10, 53,
            "A faithful PyTorch implementation of\n"
            "Andrej Karpathy's \"Let's build GPT\" lecture.\n"
            "Self-attention, multi-head, residual blocks and\n"
            "every part of a decoder-only language model.",
            fontsize=12, color=INK, linespacing=1.6, va="top")

    # technology pills
    x = 10
    for txt, c in [("PyTorch", ORANGE), ("Transformer", ACCENT),
                   ("Self-Attention", ACCENT2), ("Char-level", GREEN),
                   ("Weights & Biases", YELLOW)]:
        x = pill(ax, x, 40, txt, fc=c) + 1.5

    # bottom info box — architecture summary
    box(ax, 10, 13, 80, 22, "", fc=PANEL, ec=GRID, rounding=0.04)
    ax.text(14, 31, "Architecture at a Glance", fontsize=12, color=ACCENT,
            weight="bold")
    specs = [
        ("Dataset", "Tiny Shakespeare (~1.1 MB text)"),
        ("Tokenizer", "Character-level  ·  vocab ≈ 65"),
        ("Model", "Decoder-only Transformer  ·  6 blocks  ·  8 heads"),
        ("Embedding size", "n_embed = 256  ·  block_size = 256"),
        ("Training", "AdamW + Cosine LR  ·  Early Stopping  ·  W&B"),
    ]
    yy = 27
    for k, v in specs:
        ax.text(14, yy, k, fontsize=9.5, color=MUTED, weight="bold")
        ax.text(38, yy, v, fontsize=9.5, color=INK)
        yy -= 3.1

    ax.text(10, 7, "github.com/gocenalper/nano-gpt", fontsize=10,
            color=ACCENT, weight="bold")
    ax.text(90, 7, "2026", fontsize=10, color=MUTED, ha="right")

    pdf.savefig(fig); plt.close(fig)


# ===========================================================================
# PAGE 2 — END-TO-END PIPELINE
# ===========================================================================
def page_pipeline(pdf):
    fig, ax = new_page()
    header(ax, "OVERVIEW", "End-to-End Data Flow", 1)

    ax.text(5, 88,
            "The full pipeline, from raw text to freshly generated text. "
            "Each box maps to\na function or module in the notebook.",
            fontsize=10, color=MUTED, va="top", linespacing=1.5)

    stages = [
        ("Tiny\nShakespeare", "download_dataset()", ORANGE),
        ("Tokenizer\n(stoi / itos)", "build_tokenizer()", PINK),
        ("Train / Val\ntensors", "make_splits()", YELLOW),
        ("Batch\n(B, T)", "get_batch()", GREEN),
        ("GPT\nModel", "BigramLanguageModel", ACCENT),
        ("Generated\nText", "model.generate()", ACCENT2),
    ]
    centers = []
    y0 = 78
    for i, (title, sub, c) in enumerate(stages):
        yy = y0 - i * 11.5
        ctr = box(ax, 18, yy - 4, 30, 8, title, fc=PANEL, ec=c, tc=INK,
                  fs=11, weight="bold", lw=2.2)
        ax.text(52, yy + 0.6, sub, fontsize=10, color=c, weight="bold",
                va="center", family="DejaVu Sans Mono")
        descs = [
            "Downloads ~1.1 MB of raw text from Karpathy's char-rnn repo.",
            "Maps every unique character to an integer (encode / decode).",
            "Converts text to a single long tensor, 90% train / 10% val.",
            "Pulls (context, target) pairs from random starting points.",
            "Embedding -> N x Transformer Block -> LayerNorm -> lm_head.",
            "Autoregressive sampling generates new text character by character.",
        ]
        ax.text(52, yy - 2.4, descs[i], fontsize=8, color=MUTED, va="center")
        centers.append((ctr, c))

    for i in range(len(stages) - 1):
        y_top = 78 - i * 11.5 - 4
        y_bot = 78 - (i + 1) * 11.5 + 4
        arrow(ax, (33, y_top), (33, y_bot), color=centers[i][1], lw=2.4)

    footer(ax, 2)
    pdf.savefig(fig); plt.close(fig)


# ===========================================================================
# PAGE 3 — TOKENIZER & BATCH
# ===========================================================================
def page_data(pdf):
    fig, ax = new_page()
    header(ax, "DATA LAYER", "Tokenizer and Batch Construction", 2)

    # --- Tokenizer block ---
    ax.text(5, 88, "1 · Character-Level Tokenizer", fontsize=12,
            color=ACCENT, weight="bold")
    ax.text(5, 85, "Instead of BPE, each character is a single id. The vocab "
            "is tiny (~65), the embedding table is small, debugging is easy.",
            fontsize=9, color=MUTED, va="top")

    chars = list("Hi!")
    ids = [20, 47, 2]
    bx = 10
    for ch in chars:
        box(ax, bx, 76, 6, 6, ch, fc=PANEL, ec=PINK, fs=16, weight="bold")
        bx += 7
    arrow(ax, (31, 79), (39, 79), color=PINK)
    ax.text(35, 81.4, "encode", fontsize=8, color=PINK, ha="center")
    bx = 40
    for i in ids:
        box(ax, bx, 76, 6, 6, str(i), fc=PANEL, ec=YELLOW, fs=14,
            weight="bold", tc=YELLOW)
        bx += 7
    arrow(ax, (61, 79), (69, 79), color=YELLOW)
    ax.text(65, 81.4, "decode", fontsize=8, color=YELLOW, ha="center")
    bx = 70
    for ch in chars:
        box(ax, bx, 76, 6, 6, ch, fc=PANEL, ec=PINK, fs=16, weight="bold")
        bx += 7

    # --- Batch block ---
    ax.text(5, 67, "2 · get_batch  →  (context, target) pairs", fontsize=12,
            color=ACCENT, weight="bold")
    ax.text(5, 64,
            "A block_size-long x slice is paired with y, shifted one to the "
            "right. A single slice\ncarries T independent training examples — "
            "this is the source of the Transformer's parallel learning.",
            fontsize=9, color=MUTED, va="top", linespacing=1.5)

    seq = ['F', 'i', 'r', 's', 't', ' ', 'C', 'i']
    ax.text(8, 54, "x :", fontsize=11, color=GREEN, weight="bold", va="center")
    for j, ch in enumerate(seq[:-1]):
        box(ax, 13 + j * 9, 51, 8, 6, ch, fc=PANEL, ec=GREEN, fs=13,
            weight="bold")
    ax.text(8, 44, "y :", fontsize=11, color=ORANGE, weight="bold", va="center")
    for j, ch in enumerate(seq[1:]):
        box(ax, 13 + j * 9, 41, 8, 6, ch, fc=PANEL, ec=ORANGE, fs=13,
            weight="bold")
    for j in range(len(seq) - 1):
        arrow(ax, (17 + j * 9, 51), (17 + j * 9, 47), color=MUTED, lw=1.2,
              style="-|>")

    ax.text(50, 35,
            "prediction direction:  given  x[:, t]  the model predicts  y[:, t]",
            fontsize=8.5, color=MUTED, ha="center", style="italic")

    # --- Shape box ---
    box(ax, 10, 12, 80, 18, "", fc=PANEL, ec=GRID, rounding=0.04)
    ax.text(14, 27, "Tensor Shapes", fontsize=11, color=ACCENT,
            weight="bold")
    rows = [
        ("x, y", "(B, T)", "B = batch_size, T = block_size"),
        ("token_embedding", "(B, T, C)", "C = n_embed"),
        ("logits", "(B, T, vocab_size)", "score for each position"),
        ("loss", "scalar", "cross-entropy"),
    ]
    yy = 23
    for name, shape, note in rows:
        ax.text(14, yy, name, fontsize=9.5, color=INK, weight="bold",
                family="DejaVu Sans Mono")
        ax.text(40, yy, shape, fontsize=9.5, color=GREEN,
                family="DejaVu Sans Mono")
        ax.text(60, yy, note, fontsize=8.5, color=MUTED)
        yy -= 3.4

    footer(ax, 3)
    pdf.savefig(fig); plt.close(fig)


# ===========================================================================
# PAGE 4 — FULL MODEL ARCHITECTURE
# ===========================================================================
def page_arch(pdf):
    fig, ax = new_page()
    header(ax, "ARCHITECTURE", "BigramLanguageModel — High-Level Flow", 3)

    cx = 30
    w = 36
    layers = [
        ("idx  (B, T)", GRID, "input token ids"),
        ("Token Embedding\n+ Position Embedding", PINK, "(B, T, 256)"),
        ("Dropout", MUTED, "tok_pos_dropout"),
        ("Transformer Block × 6", ACCENT, "the heart →"),
        ("Final LayerNorm", YELLOW, "ln_f"),
        ("lm_head (Linear)", GREEN, "→ (B, T, vocab)"),
        ("logits → softmax", ACCENT2, "probability distribution"),
    ]
    y0 = 84
    gap = 11
    for i, (txt, c, note) in enumerate(layers):
        yy = y0 - i * gap
        big = (i == 3)
        h = 7.5
        box(ax, cx, yy - h / 2, w, h, txt,
            fc=(PANEL if not big else "#1b2a3a"),
            ec=c, tc=INK, fs=11 if not big else 12,
            weight="bold", lw=2.0 if not big else 2.8)
        ax.text(cx + w + 3, yy, note, fontsize=8.5, color=c, va="center")
        if i > 0:
            arrow(ax, (cx + w / 2, y0 - (i - 1) * gap - h / 2),
                  (cx + w / 2, yy + h / 2), color=layers[i - 1][1], lw=2.2)

    box(ax, 4, 38, 20, 10, "Detail on\nnext\npage →", fc=BG, ec=ACCENT,
        tc=ACCENT, fs=9, weight="bold", rounding=0.06)
    arrow(ax, (24, 43), (cx, 84 - 3 * gap), color=ACCENT, lw=1.6, ls="--")

    footer(ax, 4)
    pdf.savefig(fig); plt.close(fig)


# ===========================================================================
# PAGE 5 — TRANSFORMER BLOCK
# ===========================================================================
def page_block(pdf):
    fig, ax = new_page()
    header(ax, "ARCHITECTURE", "Transformer Block — Communicate + Compute", 4)

    ax.text(5, 88,
            "Each block has two sub-layers. Pre-LayerNorm is applied and the "
            "output of each\nsub-layer is added back to the input via a "
            "residual connection:  x = x + Dropout(SubLayer(LN(x))).",
            fontsize=9.5, color=MUTED, va="top", linespacing=1.5)

    box(ax, 38, 78, 24, 6, "input  x", fc=PANEL, ec=GRID, fs=11, weight="bold")

    box(ax, 30, 66, 40, 6, "LayerNorm (ln1)", fc=PANEL, ec=YELLOW, fs=10)
    box(ax, 24, 55, 52, 7, "Multi-Head Self-Attention",
        fc="#1b2a3a", ec=ACCENT, fs=12, weight="bold", lw=2.6)
    box(ax, 38, 47, 24, 5, "Dropout", fc=PANEL, ec=MUTED, fs=9)

    res1 = box(ax, 44, 39, 12, 5, "(+)", fc=PANEL, ec=GREEN, tc=GREEN,
               fs=14, weight="bold")

    box(ax, 30, 30, 40, 5, "LayerNorm (ln2)", fc=PANEL, ec=YELLOW, fs=10)
    box(ax, 24, 18, 52, 8,
        "Feed-Forward\nLinear(C->4C) · ReLU · Linear(4C->C)",
        fc="#2a1b3a", ec=ACCENT2, fs=10, weight="bold", lw=2.6)
    box(ax, 38, 11, 24, 4.5, "Dropout", fc=PANEL, ec=MUTED, fs=9)
    res2 = box(ax, 44, 5, 12, 4.5, "(+)", fc=PANEL, ec=GREEN, tc=GREEN,
               fs=14, weight="bold")

    seq_y = [(81, 72), (66, 62), (58.5, 52), (49.5, 44),
             (39, 35), (32.5, 26), (18, 15.5), (11, 9.5)]
    for top, bot in seq_y:
        arrow(ax, (50, top), (50, bot), color=INK, lw=1.8)

    # residual skip connections
    arrow(ax, (62, 80), (88, 80), color=GREEN, lw=2, style="-")
    arrow(ax, (88, 80), (88, 41.5), color=GREEN, lw=2, style="-")
    arrow(ax, (88, 41.5), (56, 41.5), color=GREEN, lw=2)
    ax.text(89.5, 60, "residual", fontsize=8, color=GREEN, rotation=90,
            va="center")

    arrow(ax, (44, 41.5), (12, 41.5), color=GREEN, lw=2, style="-", rad=0)
    arrow(ax, (12, 41.5), (12, 7.2), color=GREEN, lw=2, style="-")
    arrow(ax, (12, 7.2), (44, 7.2), color=GREEN, lw=2)
    ax.text(10.5, 25, "residual", fontsize=8, color=GREEN, rotation=90,
            va="center")

    footer(ax, 5)
    pdf.savefig(fig); plt.close(fig)


# ===========================================================================
# PAGE 6 — SELF-ATTENTION HEAD
# ===========================================================================
def page_attention(pdf):
    fig, ax = new_page()
    header(ax, "ARCHITECTURE", "A Single Self-Attention Head", 5)

    ax.text(5, 88,
            "Each head splits the input into three projections: Query, Key, "
            "Value. Scores are scaled,\nthe future is hidden with a causal "
            "mask, softmax turns them into weights, Values are summed.",
            fontsize=9.5, color=MUTED, va="top", linespacing=1.5)

    box(ax, 6, 70, 16, 7, "x\n(B,T,C)", fc=PANEL, ec=GRID, fs=10, weight="bold")

    qkv = [("Query\nq = Wq·x", ACCENT, 80), ("Key\nk = Wk·x", PINK, 70),
           ("Value\nv = Wv·x", ORANGE, 60)]
    for txt, c, yy in qkv:
        box(ax, 30, yy - 3.5, 18, 7, txt, fc=PANEL, ec=c, tc=INK, fs=9.5,
            weight="bold", lw=2)
        arrow(ax, (22, 73.5), (30, yy), color=c, lw=1.8)

    box(ax, 56, 70, 18, 7, "scores\nq · kᵀ", fc=PANEL, ec=ACCENT2, fs=10,
        weight="bold", lw=2)
    arrow(ax, (48, 80), (56, 74.5), color=ACCENT)
    arrow(ax, (48, 70), (56, 72), color=PINK)
    ax.text(65, 66.5, "× (head_size)^-0.5", fontsize=8, color=MUTED,
            ha="center", style="italic")

    box(ax, 56, 56, 18, 6, "causal mask\n(tril)", fc=PANEL, ec=YELLOW, fs=9.5,
        weight="bold", lw=2)
    arrow(ax, (65, 70), (65, 62), color=ACCENT2)

    box(ax, 56, 44, 18, 6, "softmax\n+ dropout", fc=PANEL, ec=GREEN, fs=9.5,
        weight="bold", lw=2)
    arrow(ax, (65, 56), (65, 50), color=YELLOW)

    box(ax, 56, 31, 18, 7, "output\nwei · v", fc="#1b2a3a", ec=ACCENT, fs=11,
        weight="bold", lw=2.6)
    arrow(ax, (65, 44), (65, 38), color=GREEN)
    arrow(ax, (39, 60), (56, 34.5), color=ORANGE, rad=-0.2)

    ax.text(5, 24, "Causal mask:  a token may only see itself and the past "
            "(future = -inf)", fontsize=9.5, color=ACCENT,
            weight="bold")
    n = 5
    gx, gy, cell = 12, 6, 3.0
    for r in range(n):
        for col in range(n):
            allowed = col <= r
            ax.add_patch(Rectangle(
                (gx + col * cell, gy + (n - 1 - r) * cell), cell - 0.3,
                cell - 0.3,
                facecolor=(GREEN if allowed else PANEL),
                edgecolor=GRID, lw=1, alpha=0.85 if allowed else 1))
            ax.text(gx + col * cell + (cell - 0.3) / 2,
                    gy + (n - 1 - r) * cell + (cell - 0.3) / 2,
                    "1" if allowed else "0", ha="center", va="center",
                    fontsize=8, color=(BG if allowed else MUTED),
                    weight="bold")
    ax.text(gx + n * cell + 3, gy + n * cell / 2,
            "row = querying position\ncol = attended position\n\n"
            "green (1) = allowed\ndark (0) = masked",
            fontsize=9, color=MUTED, va="center", linespacing=1.5)

    footer(ax, 6)
    pdf.savefig(fig); plt.close(fig)


# ===========================================================================
# PAGE 7 — TRAINING LOOP
# ===========================================================================
def page_training(pdf):
    fig, ax = new_page()
    header(ax, "TRAINING", "Training Loop and Hyperparameters", 6)

    steps = [
        ("get_batch('train')", GREEN, "draw a random batch"),
        ("logits, loss = model(x, y)", ACCENT, "forward pass + CE loss"),
        ("loss.backward()", ORANGE, "gradients"),
        ("optimizer.step()", PINK, "AdamW update"),
        ("scheduler.step()", ACCENT2, "Cosine LR decay"),
    ]
    ax.text(5, 88, "Iteration Loop", fontsize=12, color=ACCENT,
            weight="bold")
    y0 = 82
    for i, (txt, c, note) in enumerate(steps):
        yy = y0 - i * 9
        box(ax, 6, yy - 3, 40, 6, txt, fc=PANEL, ec=c, tc=INK, fs=9.5,
            weight="bold", lw=2, family="DejaVu Sans Mono")
        ax.text(48, yy, note, fontsize=8.5, color=c, va="center")
        if i > 0:
            arrow(ax, (26, y0 - (i - 1) * 9 - 3), (26, yy + 3), color=c,
                  lw=1.8)
    arrow(ax, (6, y0 - 4 * 9), (2.5, y0 - 4 * 9), color=MUTED, style="-",
          lw=1.5)
    arrow(ax, (2.5, y0 - 4 * 9), (2.5, y0), color=MUTED, style="-", lw=1.5)
    arrow(ax, (2.5, y0), (6, y0), color=MUTED, lw=1.5)
    ax.text(1.2, y0 - 18, "repeat", fontsize=8, color=MUTED, rotation=90,
            va="center")

    box(ax, 5, 24, 90, 12, "", fc=PANEL, ec=GRID, rounding=0.03)
    ax.text(9, 33, "Regularization & Monitoring", fontsize=11, color=ACCENT,
            weight="bold")
    notes = [
        ("estimate_loss", "estimates train/val loss every 500 steps (eval mode)"),
        ("Early Stopping", "stops if val loss stalls for 5 evaluations"),
        ("Cosine LR", "lr 5e-4 -> 1e-5, smooth decay"),
        ("Weights & Biases", "all metrics logged live"),
    ]
    yy = 30
    for k, v in notes:
        ax.text(9, yy, "•", fontsize=11, color=GREEN, weight="bold")
        ax.text(11, yy, k, fontsize=9.5, color=INK, weight="bold")
        ax.text(34, yy, v, fontsize=9, color=MUTED)
        yy -= 2.6

    box(ax, 5, 6, 90, 14, "", fc=PANEL, ec=ACCENT, rounding=0.03, lw=1.6)
    ax.text(9, 17.5, "HERO RUN v4 — Configuration", fontsize=11, color=ACCENT,
            weight="bold")
    hp = [("block_size", "256"), ("batch_size", "64"), ("n_embed", "256"),
          ("n_head", "8"), ("n_layer", "6"), ("dropout", "0.2"),
          ("lr", "5e-4"), ("epochs", "10000"), ("weight_decay", "1e-3")]
    for i, (k, v) in enumerate(hp):
        xx = 9 + (i % 3) * 29
        yy = 13.5 - (i // 3) * 3.2
        ax.text(xx, yy, k, fontsize=9, color=MUTED, family="DejaVu Sans Mono")
        ax.text(xx + 16, yy, v, fontsize=9, color=YELLOW, weight="bold",
                family="DejaVu Sans Mono")

    footer(ax, 7)
    pdf.savefig(fig); plt.close(fig)


# ===========================================================================
# PAGE 8 — GENERATION + SUMMARY
# ===========================================================================
def page_generate(pdf):
    fig, ax = new_page()
    header(ax, "INFERENCE", "Autoregressive Text Generation", 7)

    ax.text(5, 88,
            "model.generate() takes the last block_size tokens as context at "
            "each step, samples one\ntoken from the last position's "
            "probability distribution and appends it. The loop feeds itself.",
            fontsize=9.5, color=MUTED, va="top", linespacing=1.5)

    loop = [
        ("idx[:, -block_size:]", ACCENT, "crop the context"),
        ("logits = model(idx)", PINK, "forward pass"),
        ("logits[:, -1, :]", YELLOW, "focus on last step"),
        ("softmax → probs", GREEN, "probabilities"),
        ("multinomial sample", ORANGE, "pick 1 token"),
        ("idx = cat(idx, next)", ACCENT2, "append to sequence"),
    ]
    cx = 28
    for i, (txt, c, note) in enumerate(loop):
        yy = 78 - i * 8
        box(ax, cx, yy - 3, 44, 6, txt, fc=PANEL, ec=c, tc=INK, fs=10,
            weight="bold", lw=2, family="DejaVu Sans Mono")
        ax.text(cx + 47, yy, note, fontsize=8.5, color=c, va="center")
        if i > 0:
            arrow(ax, (cx + 22, 78 - (i - 1) * 8 - 3), (cx + 22, yy + 3),
                  color=c, lw=2)
    arrow(ax, (cx, 78 - 5 * 8), (20, 78 - 5 * 8), color=MUTED, style="-")
    arrow(ax, (20, 78 - 5 * 8), (20, 78), color=MUTED, style="-")
    arrow(ax, (20, 78), (cx, 78), color=MUTED)
    ax.text(18, 56, "max_new_tokens times", fontsize=8, color=MUTED,
            rotation=90, va="center")

    box(ax, 5, 8, 90, 20, "", fc="#1b2a3a", ec=ACCENT, rounding=0.03, lw=1.6)
    ax.text(9, 24.5, "Why It Matters", fontsize=12, color=ACCENT,
            weight="bold")
    ax.text(9, 21,
            "This tiny model contains the exact same core mechanism as modern "
            "large language models (GPT):\n"
            "token + position embedding, causal self-attention, multi-head, "
            "residual blocks and autoregressive generation.\n"
            "Apart from scale, the fundamental architecture is identical — "
            "which makes it an ideal way to learn \"GPT from scratch\".",
            fontsize=9.5, color=INK, va="top", linespacing=1.6)

    footer(ax, 8)
    pdf.savefig(fig); plt.close(fig)


def main():
    out = Path(__file__).parent / "nano-gpt-documentation.pdf"
    with PdfPages(out) as pdf:
        page_cover(pdf)
        page_pipeline(pdf)
        page_data(pdf)
        page_arch(pdf)
        page_block(pdf)
        page_attention(pdf)
        page_training(pdf)
        page_generate(pdf)
        d = pdf.infodict()
        d["Title"] = "nano-gpt — Character-Level GPT Documentation"
        d["Author"] = "gocenalper"
        d["Subject"] = "A Transformer / GPT implementation from scratch"
    print(f"PDF written: {out}")


if __name__ == "__main__":
    main()
