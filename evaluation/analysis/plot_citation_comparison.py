"""Standalone citation comparison figure. Values transcribed from the supplied reference image."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, PathPatch, Patch, FancyBboxPatch
from matplotlib.path import Path as MPath
from matplotlib.ticker import MaxNLocator

COLORS = ['#307EB6', '#F4BC31', '#ED343B', '#FA7B20', '#389F94']
DOMAINS = ['General Fairness & Bias Mitigation', 'LLM and NLP',
           'Health & Clinical AI', 'Recommender Systems',
           'Graph-based Fairness & Bias Mitigation']
LABELS = ['General Fairness\n& Bias Mitigation', 'LLMs & NLP',
          'Health & Clinical AI', 'Recommender\nSystems',
          'Graph-Based\nFairness & Bias\nMitigation']
LEGEND = ['General Fairness & Bias Mitigation', 'LLMs & NLP',
          'Health & Clinical AI', 'Recommender Systems',
          'Graph-Based Fairness & Bias Mitigation']


def icon(ax, kind, color):
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal')
    ax.axis('off')
    def line(x, y, c=color, lw=2.5):
        ax.plot(x, y, color=c, lw=lw, solid_capstyle='round', solid_joinstyle='round')
    def path(verts, codes, fill=False, c=color, lw=2.4):
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=c if fill else 'none',
                               edgecolor=c, lw=lw, capstyle='round', joinstyle='round'))
    M, L, C, Z = MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CLOSEPOLY
    if kind == 0:  # Balance scales
        line([.5, .5], [.17, .89]); line([.16, .84], [.73, .73])
        line([.33, .67], [.13, .13], lw=4)
        line([.39, .61], [.21, .21], lw=4)
        ax.add_patch(Circle((.5, .78), .055, color=color))
        for x in [.22, .78]:
            line([x-.13, x, x+.13], [.40, .72, .40], lw=1.8)
            path([(x-.14,.40),(x-.11,.22),(x+.11,.22),(x+.14,.40),(x-.14,.40)],
                 [M,C,C,C,Z], fill=True, lw=1)
    elif kind == 1:  # Brain outline and folds
        for flip in [False, True]:
            v = [(.49,.84),(.38,1.0),(.24,.90),(.26,.79),
                 (.09,.83),(.06,.62),(.16,.56),
                 (.02,.48),(.08,.29),(.22,.30),
                 (.13,.15),(.31,.06),(.39,.17),
                 (.44,.08),(.49,.17),(.49,.25),(.49,.84)]
            v = [(1-x if flip else x,y) for x,y in v]
            path(v,[M]+[C]*15+[L])
            for vv in [[(.29,.78),(.40,.73),(.38,.64),(.29,.61)],
                       [(.22,.47),(.36,.51),(.39,.40),(.30,.34)],
                       [(.39,.23),(.30,.21),(.25,.26),(.27,.32)]]:
                path([(1-x if flip else x,y) for x,y in vv],[M,C,C,C],lw=2)
    elif kind == 2:  # Heart with ECG
        path([(.5,.12),(.36,.24),(.06,.46),(.06,.68),
              (.06,.95),(.38,1.0),(.5,.76),
              (.62,1.0),(.94,.95),(.94,.68),
              (.94,.46),(.64,.24),(.5,.12)], [M]+[C]*12,fill=True,lw=0)
        line([.10,.32,.41,.49,.57,.65,.90], [.49,.49,.65,.34,.56,.49,.49],c='white',lw=2)
    elif kind == 3:  # Person, recommendation star and thumbs-up
        ax.add_patch(Circle((.25,.65),.115,facecolor=color,edgecolor='none'))
        path([(.06,.16),(.06,.54),(.44,.54),(.44,.16),(.06,.16)], [M,C,C,C,Z], fill=True,lw=0)
        angles=np.linspace(np.pi/2, np.pi/2+2*np.pi, 10,endpoint=False)
        rr=np.array([.17,.075]*5)
        ax.add_patch(Polygon(np.c_[.62+rr*np.cos(angles),.82+rr*np.sin(angles)],color=color))
        ax.add_patch(FancyBboxPatch((.50,.18),.08,.29,boxstyle='round,pad=0.005,rounding_size=0.02',facecolor=color,edgecolor='none'))
        path([(.62,.18),(.84,.18),(.90,.45),(.73,.45),(.76,.60),
              (.69,.63),(.62,.43),(.62,.18)], [M]+[L]*6+[Z], fill=True,lw=0)
    else:  # Graph nodes
        nodes=[(.22,.22),(.78,.22),(.22,.78),(.78,.78)]
        for x,y in nodes:
            line([.5,x],[.5,y],lw=4)
            ax.add_patch(Circle((x,y),.115,facecolor=color,edgecolor='none'))
        ax.add_patch(Circle((.5,.5),.12,facecolor=color,edgecolor='none'))


# Values transcribed from the user's citation figure, in the category order above.
# These are citation totals, not paper counts; they are not computed from papers.csv.
INTERNAL_CITATIONS = np.array([1052, 261, 240, 95, 42])
GLOBAL_CITATIONS = np.array([29555, 13080, 18400, 2195, 1504])


def main():
    from matplotlib.ticker import StrMethodFormatter

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    fig = plt.figure(figsize=(17, 6.8), facecolor='white')
    left = fig.add_axes([.215, .16, .31, .65])
    right = fig.add_axes([.625, .16, .35, .65])
    panels = [
        (left, INTERNAL_CITATIONS, '(a)  Internal citations', 'Within the 692-paper corpus'),
        (right, GLOBAL_CITATIONS, '(b)  Global citations', 'Across academic literature via OpenAlex'),
    ]
    for ax, values, title, subtitle in panels:
        ax.barh(np.arange(5), values, color=COLORS, height=.70, edgecolor='none')
        ax.set(ylim=(4.55, -.55), xlim=(0, values.max() * 1.24),
               xlabel='Total citations', yticks=[])
        ax.xaxis.label.set_size(14)
        ax.xaxis.labelpad = 12
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.tick_params(axis='x', labelsize=11)
        ax.grid(axis='x', linestyle='--', linewidth=.7, alpha=.22)
        ax.set_axisbelow(True)
        ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=19, weight='bold')
        ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=12, color='#444444')
        for k, value in enumerate(values):
            ax.text(value + values.max() * .025, k, f'{value:,}',
                    va='center', ha='left', fontsize=14, weight='bold')

    # Shared row labels and matching vector icons align both citation panels.
    for k, (label, color) in enumerate(zip(LABELS, COLORS)):
        yy = fig.transFigure.inverted().transform(left.transData.transform((0, k)))[1]
        fig.text(.070, yy, label, va='center', fontsize=13, linespacing=1.2)
        icon_ax = fig.add_axes([.019, yy - .045, .040, .090])
        icon(icon_ax, k, color)

    output = ROOT / 'figures'
    output.mkdir(exist_ok=True)
    stem = 'Internal_and_global_citations_by_domain'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{stem}.{ext}', dpi=600, facecolor='white')
    fig.savefig(output / f'{stem}_preview.png', dpi=120, facecolor='white')
    plt.close(fig)
    print(f'Saved PNG, PDF and SVG to {output / stem}')


if __name__ == '__main__':
    main()

