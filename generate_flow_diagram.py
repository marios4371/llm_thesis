"""
Generate a professional, minimal flow diagram for the MAS + SHT pipeline.
Version 2: Cleaner layout, no overlaps.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ─── Setup ───
fig, ax = plt.subplots(1, 1, figsize=(18, 30))
ax.set_xlim(0, 18)
ax.set_ylim(0, 30)
ax.axis('off')
fig.patch.set_facecolor('#FAFBFC')

# ─── Colors ───
C = {
    'blue':       '#2563EB',
    'purple':     '#7C3AED',
    'green':      '#059669',
    'amber':      '#D97706',
    'red':        '#DC2626',
    'slate':      '#475569',
    'text':       '#1E293B',
    'text_light': '#64748B',
    'border':     '#CBD5E1',
    'white':      '#FFFFFF',
    'light_blue': '#DBEAFE',
    'light_purple':'#F5F3FF',
    'light_green':'#ECFDF5',
    'light_red':  '#FEF2F2',
    'light_amber':'#FFFBEB',
    'bg':         '#F8FAFC',
}


def rbox(x, y, w, h, label, sub=None, color=C['blue'], bg=C['white'],
         fs=10, bar=True):
    """Rounded box with optional colored top bar."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.12,rounding_size=0.25",
                         facecolor=bg, edgecolor=color, linewidth=1.6, zorder=3)
    ax.add_patch(box)
    if bar:
        bh = 0.28
        top_bar = FancyBboxPatch((x - w/2, y + h/2 - bh), w, bh,
                                  boxstyle="round,pad=0,rounding_size=0.25",
                                  facecolor=color, edgecolor='none', zorder=4,
                                  clip_on=True)
        ax.add_patch(top_bar)
        # cover bottom rounded corners of bar
        ax.add_patch(plt.Rectangle((x - w/2, y + h/2 - bh), w, bh * 0.5,
                                    facecolor=color, edgecolor='none', zorder=4))
    if sub:
        ax.text(x, y + 0.05, label, fontsize=fs, color=C['text'],
                ha='center', va='center', zorder=5, fontweight='bold',
                fontfamily='sans-serif')
        ax.text(x, y - 0.28, sub, fontsize=fs - 2.5, color=C['text_light'],
                ha='center', va='center', zorder=5, fontfamily='sans-serif',
                style='italic')
    else:
        ax.text(x, y, label, fontsize=fs, color=C['text'],
                ha='center', va='center', zorder=5, fontweight='bold',
                fontfamily='sans-serif')


def diamond(x, y, sw, sh, label, color=C['amber']):
    """Decision diamond."""
    pts = [(x, y + sh), (x + sw, y), (x, y - sh), (x - sw, y)]
    d = plt.Polygon(pts, facecolor=C['light_amber'], edgecolor=color,
                    linewidth=1.6, zorder=3)
    ax.add_patch(d)
    ax.text(x, y, label, fontsize=7.5, color=C['text'], ha='center',
            va='center', zorder=5, fontweight='bold', fontfamily='sans-serif')


def arr(x1, y1, x2, y2, color=C['slate'], lw=1.4, dash=False):
    """Arrow."""
    style = (0, (4, 3)) if dash else '-'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=style), zorder=2)


def label_on_arrow(x, y, text, color=C['slate'], fs=7.5):
    """Small label near an arrow."""
    ax.text(x, y, text, fontsize=fs, color=color, ha='center', va='center',
            fontweight='bold', fontfamily='sans-serif',
            bbox=dict(boxstyle='round,pad=0.12', facecolor=C['white'],
                      edgecolor='none', alpha=0.95), zorder=6)


def badge(x, y, text, color=C['blue']):
    """API call badge."""
    ax.text(x, y, text, fontsize=6.5, color=C['white'], ha='center',
            va='center', zorder=6, fontweight='bold', fontfamily='sans-serif',
            bbox=dict(boxstyle='round,pad=0.18', facecolor=color,
                      edgecolor='none', alpha=0.9))


def phase_label(y, num, color):
    """Left-side phase indicator."""
    ax.text(0.6, y, f'PHASE\n{num}', fontsize=7, color=color,
            fontweight='bold', fontfamily='sans-serif', ha='center',
            va='center')
    ax.plot([1.1, 1.1], [y - 0.7, y + 0.7], color=color, linewidth=3,
            solid_capstyle='round', zorder=1)


# ══════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════

ax.text(9, 29.3, 'MAS + Structured Hypothesis Testing (SHT)',
        fontsize=19, fontweight='bold', color=C['text'], ha='center',
        fontfamily='sans-serif')
ax.text(9, 28.85, 'Multi-Agent Math Solver  ·  Architecture Flow  ·  v7.0',
        fontsize=10, color=C['text_light'], ha='center',
        fontfamily='sans-serif')
ax.plot([2, 16], [28.55, 28.55], color=C['border'], linewidth=0.8)


# ══════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════

rbox(9, 28.0, 4.5, 0.75, 'Math Problem', 'from dataset + distractors',
     color=C['slate'], bg=C['bg'], fs=11)

arr(9, 27.62, 9, 27.1)


# ══════════════════════════════════════════════════════════════════
# PHASE 1: THREE PARALLEL AGENTS
# ══════════════════════════════════════════════════════════════════

phase_label(26.3, '1', C['blue'])

# Branch arrows from center
arr(7.5, 27.1, 4.5, 26.75, C['slate'], 1.2)
arr(9, 27.1, 9, 26.75, C['slate'], 1.2)
arr(10.5, 27.1, 13.5, 26.75, C['slate'], 1.2)

# Baseline
rbox(4.5, 26.2, 3.6, 0.85, 'Baseline', 'Zero-shot · temp=0.1',
     color=C['slate'], fs=9.5)
badge(6.5, 25.95, 'API ×1', C['slate'])

# Mathematician
rbox(9, 26.2, 3.6, 0.85, 'Mathematician', 'JSON Blueprint · temp=0.0',
     color=C['blue'], fs=9.5)
badge(11.0, 25.95, 'API ×1', C['blue'])

# Programmer
rbox(13.5, 26.2, 3.6, 0.85, 'Programmer', 'Code gen+exec · temp=0.05',
     color=C['blue'], fs=9.5)
badge(15.5, 25.95, 'API ×1-3', C['blue'])

# Blueprint arrow
arr(10.8, 26.2, 11.7, 26.2, C['blue'], 1.2)
label_on_arrow(11.25, 26.42, 'blueprint', C['blue'], 6.5)

# Output labels
arr(4.5, 25.78, 4.5, 25.2, C['slate'], 1.0)
ax.text(4.5, 25.0, 'base_ans', fontsize=8, color=C['slate'], ha='center',
        fontfamily='monospace', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.12', facecolor=C['bg'],
                  edgecolor=C['border']))

arr(13.5, 25.78, 13.5, 25.2, C['blue'], 1.0)
ax.text(13.5, 25.0, 'primary_ans', fontsize=8, color=C['blue'], ha='center',
        fontfamily='monospace', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.12', facecolor=C['light_blue'],
                  edgecolor=C['border']))


# ══════════════════════════════════════════════════════════════════
# SHT BACKGROUND
# ══════════════════════════════════════════════════════════════════

sht_bg = FancyBboxPatch((1.5, 11.3), 15, 13.0,
                         boxstyle="round,pad=0.35,rounding_size=0.4",
                         facecolor=C['light_purple'], edgecolor=C['purple'],
                         linewidth=1.2, linestyle=(0, (5, 3)),
                         alpha=0.25, zorder=0)
ax.add_patch(sht_bg)
ax.text(2.5, 24.0, 'STRUCTURED HYPOTHESIS TESTING (SHT)', fontsize=8.5,
        color=C['purple'], fontweight='bold', fontfamily='sans-serif',
        alpha=0.7)


# ══════════════════════════════════════════════════════════════════
# PHASE 2: CONFIDENCE GATE
# ══════════════════════════════════════════════════════════════════

phase_label(23.0, '2', C['purple'])

# Converge arrows to gate
arr(4.5, 24.8, 7.5, 23.35, C['slate'], 1.0)
arr(13.5, 24.8, 10.5, 23.35, C['blue'], 1.0)

diamond(9, 23.0, 1.5, 0.45, 'Confidence Gate', C['amber'])

# Criteria
criteria = ("1. answer = \"unknown\"?\n"
            "2. primary != baseline?\n"
            "3. repair loop exhausted?\n"
            "4. negative answer?\n"
            "5. extreme magnitude?")
ax.text(14.2, 23.0, criteria, fontsize=6.8, color=C['text_light'],
        ha='left', va='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['white'],
                  edgecolor=C['border'], linewidth=0.7))

# CONFIDENT → right
arr(10.5, 23.0, 16.0, 23.0, C['green'], 1.8)
label_on_arrow(13.2, 23.3, 'CONFIDENT → skip SHT', C['green'], 8)

# Return primary box (right side)
rbox(16.5, 21.5, 1.8, 0.65, 'Return', 'primary', color=C['green'],
     bg=C['light_green'], fs=8.5, bar=False)
arr(16.5, 22.55, 16.5, 21.82, C['green'], 1.2)

# UNCERTAIN → down
arr(9, 22.55, 9, 21.8, C['amber'], 1.8)
label_on_arrow(9.9, 22.15, 'UNCERTAIN', C['amber'], 8)


# ══════════════════════════════════════════════════════════════════
# PHASE 3: HYPOTHESIS GENERATION
# ══════════════════════════════════════════════════════════════════

phase_label(21.0, '3', C['purple'])

rbox(9, 21.2, 5.5, 0.9, 'Hypothesis Generator',
     'Generate 2 alternative blueprints', color=C['purple'], fs=10)
badge(12.0, 20.95, 'API ×1', C['purple'])

# Strategy tags
strats = [('Arithmetic', '#3B82F6'), ('Algebraic', '#8B5CF6'),
          ('Unit-Rate', '#06B6D4'), ('Backwards', '#F59E0B'),
          ('Partition', '#10B981')]
for i, (name, col) in enumerate(strats):
    bx = 3.2 + i * 2.4
    ax.text(bx, 20.4, name, fontsize=6, color=col, ha='center',
            fontweight='bold', fontfamily='sans-serif',
            bbox=dict(boxstyle='round,pad=0.12', facecolor=C['white'],
                      edgecolor=col, linewidth=0.7))

# Arrows to two programmers
arr(7.5, 20.75, 5.5, 20.0, C['purple'], 1.2)
arr(10.5, 20.75, 12.5, 20.0, C['purple'], 1.2)


# ══════════════════════════════════════════════════════════════════
# PHASE 4: ALTERNATIVE EXECUTION
# ══════════════════════════════════════════════════════════════════

phase_label(19.2, '4', C['purple'])

rbox(5.5, 19.4, 3.3, 0.85, 'Programmer #1',
     'alt_blueprint_1 · 1 attempt', color=C['purple'], bg='#FAF5FF', fs=9)
badge(7.3, 19.15, 'API ×1', C['purple'])

rbox(12.5, 19.4, 3.3, 0.85, 'Programmer #2',
     'alt_blueprint_2 · 1 attempt', color=C['purple'], bg='#FAF5FF', fs=9)
badge(14.3, 19.15, 'API ×1', C['purple'])

# Arrows to candidates
arr(5.5, 18.98, 9, 18.25, C['purple'], 1.0)
arr(12.5, 18.98, 9, 18.25, C['purple'], 1.0)

# Candidates
cbox = FancyBboxPatch((3.5, 17.75), 11, 0.55,
                       boxstyle="round,pad=0.08,rounding_size=0.18",
                       facecolor=C['white'], edgecolor=C['purple'],
                       linewidth=1.2, zorder=3)
ax.add_patch(cbox)
ax.text(9, 18.02, 'candidates = [ primary,  baseline,  alt_1,  alt_2 ]',
        fontsize=8, color=C['purple'], ha='center', va='center',
        zorder=5, fontfamily='monospace', fontweight='bold')


# ══════════════════════════════════════════════════════════════════
# PHASE 5: TRIAGE
# ══════════════════════════════════════════════════════════════════

phase_label(16.6, '5', C['purple'])

arr(9, 17.75, 9, 17.1, C['purple'], 1.2)

diamond(9, 16.7, 1.3, 0.4, 'Triage (voting)', C['purple'])

# Triage detail
triage_text = ("Group by answer value\n"
               "(tolerance < 0.001)\n"
               "Majority ≥ 2?")
ax.text(13.8, 16.7, triage_text, fontsize=6.8, color=C['text_light'],
        ha='left', va='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=C['white'],
                  edgecolor=C['border'], linewidth=0.7))

# MAJORITY → left
arr(7.7, 16.7, 5.5, 16.7, C['green'], 1.8)
label_on_arrow(6.6, 16.95, 'MAJORITY', C['green'], 8)

rbox(4.0, 16.7, 2.2, 0.6, 'Return', 'majority ans',
     color=C['green'], bg=C['light_green'], fs=8.5, bar=False)

# NO MAJORITY → down
arr(9, 16.3, 9, 15.6, C['amber'], 1.8)
label_on_arrow(9.9, 15.95, 'NO MAJORITY', C['amber'], 7.5)


# ══════════════════════════════════════════════════════════════════
# PHASE 6: JUDGE
# ══════════════════════════════════════════════════════════════════

phase_label(14.8, '6', C['red'])

rbox(9, 15.0, 5.0, 0.95, 'Judge Agent',
     'Evaluate reasoning quality', color=C['red'], fs=10)
badge(11.7, 14.72, 'API ×1', C['red'])

# Judge criteria
judge_text = ("1. Code execution success\n"
              "2. Mathematical correctness\n"
              "3. Completeness\n"
              "4. Cross-strategy agreement\n"
              "5. Simplicity")
ax.text(13.8, 15.0, judge_text, fontsize=6.5, color=C['text_light'],
        ha='left', va='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=C['white'],
                  edgecolor=C['border'], linewidth=0.7))

arr(9, 14.52, 9, 13.95, C['red'], 1.5)


# ══════════════════════════════════════════════════════════════════
# FINAL ANSWER (merge point)
# ══════════════════════════════════════════════════════════════════

rbox(9, 13.5, 4.5, 0.7, 'Final Answer', 'selected by SHT pipeline',
     color=C['green'], bg=C['light_green'], fs=11)

# Merge confident_skip path
arr(16.5, 21.17, 16.5, 13.5, C['green'], 1.0, dash=True)
arr(16.5, 13.5, 11.25, 13.5, C['green'], 1.0, dash=True)

# Merge majority path
arr(4.0, 16.4, 4.0, 13.5, C['green'], 1.0, dash=True)
arr(4.0, 13.5, 6.75, 13.5, C['green'], 1.0, dash=True)


# ══════════════════════════════════════════════════════════════════
# FALLBACK CHECK
# ══════════════════════════════════════════════════════════════════

arr(9, 13.15, 9, 12.55, C['slate'], 1.2)

diamond(9, 12.2, 1.3, 0.35, 'answer = "unknown"?', C['red'])

# YES → fallback
arr(7.7, 12.2, 5.5, 12.2, C['red'], 1.5)
label_on_arrow(6.6, 12.42, 'YES', C['red'], 8)
rbox(4.2, 12.2, 2.0, 0.55, 'Fallback', 'use base_ans',
     color=C['red'], bg=C['light_red'], fs=8.5, bar=False)

# NO → continue
arr(9, 11.85, 9, 11.3, C['green'], 1.5)
label_on_arrow(9.5, 11.55, 'NO', C['green'], 8)


# ══════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════

rbox(9, 10.8, 5.0, 0.75, 'Pipeline Output', 'CSV + Performance Report',
     color=C['slate'], bg=C['bg'], fs=10)

# CSV columns
csv = ("id · dataset · baseline_correct · mas_correct · baseline_ans · mas_ans\n"
       "sht_triggered · sht_triage_result · sht_winning_strategy · sht_api_calls")
ax.text(9, 10.05, csv, fontsize=6.5, color=C['text_light'],
        ha='center', va='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['white'],
                  edgecolor=C['border'], linewidth=0.7))


# ══════════════════════════════════════════════════════════════════
# REPORT METRICS
# ══════════════════════════════════════════════════════════════════

metrics = ("Report Metrics:  Baseline Acc · MAS+SHT Acc · Improvement\n"
           "SHT Trigger Rate · Rescue Count · Damage Count · Triage Breakdown")
ax.text(9, 9.35, metrics, fontsize=6.5, color=C['text_light'],
        ha='center', va='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['white'],
                  edgecolor=C['border'], linewidth=0.7))


# ══════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════

ly = 8.3
ax.plot([2, 16], [ly + 0.3, ly + 0.3], color=C['border'], linewidth=0.8)
ax.text(9, ly + 0.55, 'API CALL SUMMARY PER PROBLEM',
        fontsize=9, color=C['text'], ha='center', fontweight='bold',
        fontfamily='sans-serif')

scenarios = [
    ('Easy\n(confident skip)', '3 calls · ~15s', C['green']),
    ('Medium\n(majority found)', '6 calls · ~30s', C['blue']),
    ('Hard\n(judge needed)', '7 calls · ~35s', C['amber']),
    ('Worst case\n(all retries + judge)', '10 calls · ~50s', C['red']),
]
for i, (label, detail, col) in enumerate(scenarios):
    bx = 2.8 + i * 3.5
    ax.plot(bx - 0.9, ly - 0.15, 's', color=col, markersize=8, zorder=3)
    ax.text(bx, ly - 0.1, label, fontsize=7, color=C['text'],
            ha='center', va='center', fontweight='bold',
            fontfamily='sans-serif')
    ax.text(bx, ly - 0.55, detail, fontsize=7, color=col,
            ha='center', va='center', fontfamily='sans-serif',
            fontweight='bold')

# Color legend
cly = 7.2
ax.plot([2, 16], [cly + 0.25, cly + 0.25], color=C['border'], linewidth=0.5)
legend_items = [
    (C['blue'], 'Core MAS Pipeline'),
    (C['purple'], 'SHT Components'),
    (C['green'], 'Confident / Success'),
    (C['amber'], 'Decision / Uncertain'),
    (C['red'], 'Judge / Fallback'),
    (C['slate'], 'I/O / Baseline'),
]
for i, (col, label) in enumerate(legend_items):
    bx = 2.0 + (i % 3) * 5.0
    by = cly - 0.05 if i < 3 else cly - 0.45
    ax.plot(bx, by, 's', color=col, markersize=7, zorder=3)
    ax.text(bx + 0.3, by, label, fontsize=7, color=C['text_light'],
            ha='left', va='center', fontfamily='sans-serif')


# ── Save ──
plt.tight_layout(pad=0.3)
plt.savefig('/home/user/llm_thesis/sht_flow_diagram.png', dpi=200,
            bbox_inches='tight', facecolor='#FAFBFC', edgecolor='none')
print("Saved: sht_flow_diagram.png")
