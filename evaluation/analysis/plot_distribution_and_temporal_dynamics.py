"""Reference-style scientific figure with original vector category icons."""
import os
import csv
import re
import unicodedata
from collections import Counter
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


def normalize_domain(value):
    """Match known category aliases without changing source records."""
    value = unicodedata.normalize('NFKC', value).casefold()
    value = value.replace('&', ' and ')
    value = re.sub(r'[‐‑–—-]', ' ', value)
    value = re.sub(r'\bllms\b', 'llm', value)
    return ' '.join(value.split())


csv_path = ROOT / 'papers.csv'
with csv_path.open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
if not rows:
    raise ValueError(f'No paper records found in {csv_path}')
domain_lookup = {normalize_domain(name): k for k, name in enumerate(DOMAINS)}
records = []
for row_number, row in enumerate(rows, start=2):
    domain = normalize_domain(row.get('Domain') or '')
    if domain not in domain_lookup:
        raise ValueError(f'CSV row {row_number}: unknown Domain {row.get("Domain")!r}. '
                         f'Expected one of: {", ".join(LEGEND)}')
    try:
        year = int(row['Year'].strip())
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f'CSV row {row_number}: invalid Year {row.get("Year")!r}') from exc
    records.append((year, domain_lookup[domain]))
years=np.arange(min(y for y,_ in records),max(y for y,_ in records)+1)
cross=Counter(records)
data=np.array([[cross[int(y),k] for y in years] for k in range(5)])
counts=data.sum(axis=1)
totals=data.sum(axis=0)
assert counts.sum()==totals.sum()==len(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(20.76,7.7),facecolor='white')
ax1=fig.add_axes([.153,.265,.290,.575])
ax2=fig.add_axes([.565,.29,.425,.55])
fig.text(.018,.935,'(a)  Distribution of papers by research category',fontsize=19,weight='bold')
fig.text(.516,.935,'(b)  Annual publication trend by research category',fontsize=19,weight='bold')
fig.text(.049,.883,f'N = {len(rows)} papers',fontsize=17)

ax1.barh(np.arange(5), counts, color=COLORS, height=.72)
ax1.set(ylim=(4.55,-.55),xlim=(0,counts.max()*1.12),xlabel='Number of papers')
ax1.xaxis.label.set_size(15)
ax1.xaxis.labelpad=12
ax1.set_yticks([])
ax1.tick_params(axis='x',labelsize=13,length=5)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
for k,(n,label,color) in enumerate(zip(counts,LABELS,COLORS)):
    yy=fig.transFigure.inverted().transform(ax1.transData.transform((0,k)))[1]
    fig.text(.057,yy,label,va='center',ha='left',fontsize=14,linespacing=1.18)
    iax=fig.add_axes([.016,yy-.043,.033,.086])
    icon(iax,k,color)
    ax1.text(n+counts.max()*.02,k,f'{n} ({n/len(rows)*100:.1f}%)',va='center',fontsize=15,weight='bold',clip_on=False)

bottom=np.zeros(len(years))
for k in [2,0,1,3,4]:
    ax2.bar(years,data[k],bottom=bottom,color=COLORS[k],width=.80,linewidth=0)
    bottom+=data[k]
for year,n in zip(years,totals):
    if n:
        ax2.text(year,n+totals.max()*.02,str(n),ha='center',va='bottom',fontsize=13,weight='bold')
ax2.set(xlim=(years.min()-.6,years.max()+.6),ylim=(0,totals.max()*1.16),xlabel='Year',ylabel='Number of papers')
ax2.set_xticks(years)
ax2.set_xticklabels([f'{year}*' if year == 2026 else str(year) for year in years],
                    rotation=45 if len(years)>12 else 0,
                    ha='right' if len(years)>12 else 'center')
ax2.tick_params(labelsize=12)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
ax2.xaxis.label.set_size(15); ax2.yaxis.label.set_size(15)
ax2.xaxis.labelpad=15
ax2.grid(axis='y',linestyle='--',alpha=.25)
ax2.set_axisbelow(True)

order=[2,0,1,3,4]
handles=[Patch(facecolor=COLORS[k],label=LEGEND[k]) for k in order]
legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.043),ncol=5,
                  fontsize=12,frameon=True,fancybox=True,borderpad=.9,
                  handlelength=1.7,handleheight=1.3,columnspacing=1.5)
legend.get_frame().set_edgecolor('#B9BDC2')
out=ROOT/'figures'
out.mkdir(exist_ok=True)
output_name = 'Domain_distribution_and_temporal_dynamics'
for ext in ['png','pdf','svg']:
    fig.savefig(out/f'{output_name}.{ext}',dpi=600,facecolor='white')
fig.savefig(out/f'{output_name}_preview.png',dpi=110,facecolor='white')
print('Saved PNG, PDF, SVG and preview to',out)
print(f'Source: {csv_path}; N = {len(rows)} papers')
print('Category totals:', dict(zip(LEGEND, map(int, counts))))
