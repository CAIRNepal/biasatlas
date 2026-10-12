"""Country participation and citation impact from the saved BiasAtlas data."""
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


def main():
    import csv
    import textwrap
    from collections import defaultdict
    from statistics import median
    from matplotlib.ticker import StrMethodFormatter

    source = ROOT / 'source_data' / 'biasatlas'
    def read_csv(path):
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    corpus = read_csv(source / 'evaluation/analysis/papers_new.csv')
    enriched = read_csv(source / 'openalex_enriched.csv')
    by_id = {row['SN']: row for row in corpus}
    assert len(by_id) == len(corpus)
    assert len({row['SN'] for row in enriched}) == len(enriched)
    assert set(by_id) == {row['SN'] for row in enriched}
    records = []
    country_values = defaultdict(list)
    skip = {'', 'nan', 'none', 'unknown', 'global', 'n/a', 'na'}
    for entry in enriched:
        paper = by_id[entry['SN']]
        # Match the original analysis: missing citation counts are treated as zero.
        count = int(float(entry['cited_by_count'] or 0))
        records.append({**paper, 'citations': count})
        countries = {country.strip() for author, country in
                     zip(paper['Authors'].split(';'), paper['Author Regions'].split(';'))
                     if author.strip() and country.strip().lower() not in skip}
        for country in countries:
            country_values[country].append(count)
    eligible = [(name, len(values), median(values))
                for name, values in sorted(country_values.items()) if len(values) >= 10]
    # Preserve the original ranking: median order breaks ties in paper counts.
    by_median = sorted(eligible, key=lambda row: -row[2])
    by_count = sorted(by_median, key=lambda row: -row[1])[:15]
    by_median = by_median[:15]
    top = sorted(records, key=lambda row: -row['citations'])[:15]
    domain_index = {name: i for i, name in enumerate(LEGEND)}

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(18, 16), facecolor='white')
    ax_count = fig.add_axes([.13, .685, .335, .235])
    ax_median = fig.add_axes([.63, .685, .335, .235])
    for ax, rows, col, color, title, xlabel in [
        (ax_count, by_count, 1, '#737B83', '(a)  Papers by country', 'Number of papers'),
        (ax_median, by_median, 2, '#737B83', '(b)  Median citations by country', 'Median global citations'),
    ]:
        values = [row[col] for row in rows]
        ax.barh(range(len(rows)), values, color=color, height=.68)
        ax.set(yticks=range(len(rows)), yticklabels=[row[0] for row in rows],
               ylim=(len(rows)-.5, -.5), xlim=(0, max(values)*1.17), xlabel=xlabel)
        ax.tick_params(axis='y', length=0, pad=8, labelsize=11)
        ax.xaxis.labelpad=10
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(axis='x', linestyle='--', alpha=.22)
        ax.set_axisbelow(True)
        for y, value in enumerate(values):
            ax.text(value+max(values)*.025, y, f'{value:,.0f}' if value == int(value) else f'{value:,.1f}',
                    va='center', fontsize=11, weight='bold')
        ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=18, weight='bold')
        ax.text(0, 1.05, f'{len(rows)} countries with at least 10 papers',
                transform=ax.transAxes, fontsize=11, color='#444444')

    fig.text(.025, .622, '(c)  Most-cited papers', fontsize=19, weight='bold')
    fig.text(.025, .599, 'Top 15 papers in the 692-paper corpus · Global citations via OpenAlex',
             fontsize=12, color='#444444')
    ax_papers = fig.add_axes([.57, .105, .395, .465])
    values = [row['citations'] for row in top]
    ax_papers.barh(range(15), values,
                   color=[COLORS[domain_index[row['Domain']]] for row in top], height=.65)
    ax_papers.set(ylim=(14.5, -.5), xlim=(0, max(values)*1.16), yticks=[], xlabel='Global citations')
    ax_papers.xaxis.labelpad=10
    ax_papers.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax_papers.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax_papers.grid(axis='x', linestyle='--', alpha=.22)
    ax_papers.set_axisbelow(True)
    for y, row in enumerate(top):
        value = row['citations']
        ax_papers.text(value+max(values)*.02, y, f'{value:,}', va='center', fontsize=12, weight='bold')
        yy = fig.transFigure.inverted().transform(ax_papers.transData.transform((0,y)))[1]
        title = textwrap.fill(row['Paper Title'], width=70, break_long_words=False,
                              break_on_hyphens=False)
        fig.text(.066, yy, title, va='center', fontsize=11, linespacing=1.2)
        iax = fig.add_axes([.025, yy-.0115, .024, .023])
        kind = domain_index[row['Domain']]
        icon(iax, kind, COLORS[kind])
    handles = [Patch(facecolor=color, label=label) for color, label in zip(COLORS, LEGEND)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.020),
               ncol=3, frameon=False, fontsize=11, columnspacing=2,
               title='Research domains · Panel (c) only', title_fontsize=11)
    output = ROOT/'figures'
    output.mkdir(exist_ok=True)
    stem = 'Country_participation_and_citation_impact'
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'{stem}.{ext}', dpi=600, facecolor='white')
    fig.savefig(output/f'{stem}_preview.png', dpi=110, facecolor='white')
    plt.close(fig)
    print(f'Saved {stem}: PNG, PDF, SVG. {len(records)} papers; {len(eligible)} eligible countries.')
    print('Country participation counts each paper once per distinct valid author country.')


if __name__ == '__main__':
    main()
