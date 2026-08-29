"""
# Created by valler at 17/08/2026
Feature: the project's chart style, in one place.

Every notebook used to redefine `tidy()`, the ink colours and the save logic, which is how figures
drift apart. doc/00_project_conventions.md §6 says styling comes from `params.STYLE`; this module is the code that
applies it.

    import plotting as plot
    fig, ax = plt.subplots()
    plot.tidy(ax, title='...', xlabel='...')
    plot.save(fig, 'coherent_ggi', 'A_scatter')
"""
import matplotlib.pyplot as plt
import numpy as np

import params

STYLE = params.STYLE
PALETTE = STYLE['colors']

# Recessive ink. Data carries the colour; frames, grids and labels stay out of the way.
INK = '#222222'
MUTED = '#666666'
GRID = '#d9d9d9'

# Reserved, outside the categorical order: a mark that says "this value is degenerate", not
# "this is series 4". Used for the zero-floor rings in the GGI validation (D34).
FLAG = '#c0392b'


def tidy(ax, title=None, xlabel=None, ylabel=None, grid=None):
    """
    the project's axis treatment: recessive frame, muted labels, no top/right spine

    Args:
        ax: the axes to style.
        title: set as a left-aligned bold title if given.
        xlabel, ylabel: axis labels, in muted ink.
        grid: 'x', 'y', 'both' or None. Grids are a reading aid for magnitude, so they belong on
            the value axis only — passing 'both' on a bar chart is usually a mistake.
    """
    if title:
        ax.set_title(title, fontsize=STYLE['label_fs'], color=INK, fontweight='bold', loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=STYLE['tick_fs'], color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=STYLE['tick_fs'], color=MUTED)
    ax.tick_params(labelsize=STYLE['tick_fs'], colors=MUTED, length=3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return ax


def suptitle(fig, text, y=0.98):
    """figure title in the project's treatment, left-aligned with the first panel"""
    fig.suptitle(text, fontsize=STYLE['title_fs'], color=INK, fontweight='bold',
                 x=0.02, ha='left', y=y)


def legend(fig, handles, ncol=None, y=-0.04):
    """
    one figure-level legend below the panels

    Identity is never colour alone (CONVENTIONS §6): callers build handles that carry marker shape
    as well, so the categories survive a greyscale print.
    """
    fig.legend(handles=handles, loc='lower center', ncol=ncol or len(handles), frameon=False,
               fontsize=STYLE['tick_fs'], labelcolor=MUTED, bbox_to_anchor=(0.5, y))


def swatch(colour, label, marker='s', size=10):
    """a legend handle: coloured mark plus label, for categories that are not lines"""
    return plt.Line2D([], [], marker=marker, linestyle='None', markersize=size,
                      markerfacecolor=colour, markeredgecolor='white', label=label)


def save(fig, stage, name):
    """
    write a figure to `outputs/fig/<stage>/<name>.png` when params.STYLE['save'] is on

    The stage folder is the same one the producing notebook sits in, so a figure's path names the
    step that made it.
    """
    if not STYLE['save']:
        return None
    path = params.fig_dir(stage) / f'{name}.png'
    fig.savefig(path, dpi=STYLE['dpi'], bbox_inches='tight')
    return path


def grouped_bars(ax, categories, series, values, colours=None, width=None,
                 fmt='{:.3f}', annotate=True, counts=None):
    """
    grouped bar chart with value labels — the form for comparing a metric across a few groups

    Args:
        ax: target axes.
        categories: x positions, one per group (e.g. the outcomes).
        series: the grouped series names, in fixed order (e.g. the model variants).
        values: {series_name: [value per category]}.
        colours: {series_name: hex}; defaults to the project order.
        counts: optional {series_name: [n per category]}, printed under each value. Sample size is
            the first thing a reader needs when a bar rests on four observations.

    Labels sit above a positive bar and below a negative one, so a metric that can go negative
    (R2 against the 1:1 line) still reads.
    """
    colours = colours or dict(zip(series, PALETTE))
    width = width or 0.8 / len(series)
    x = np.arange(len(categories))

    for k, name in enumerate(series):
        pos = x + (k - (len(series) - 1) / 2) * width
        vals = np.asarray(values[name], dtype=float)
        ax.bar(pos, vals, width * 0.88, color=colours[name], edgecolor='white',
               linewidth=1.2, zorder=3)
        if not annotate:
            continue
        for i, (p, v) in enumerate(zip(pos, vals)):
            if not np.isfinite(v):
                continue
            text = fmt.format(v)
            if counts is not None:
                text += f'\nn={counts[name][i]}'
            above = v >= 0
            ax.annotate(text, (p, v), textcoords='offset points',
                        xytext=(0, 4 if above else -4), ha='center',
                        va='bottom' if above else 'top',
                        fontsize=STYLE['tick_fs'] - 1, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    return ax


def beeswarm(ax, values, centre=0.0, width=0.34, radius=None, colour=None, size=22,
             alpha=0.75, edgecolor='white', linewidth=0.3, zorder=3, **kwargs):
    """
    a one-dimensional swarm of points, spread sideways only to stop them overlapping

    The form for "here is every value in a distribution", where a box plot would hide how many
    points there are and a histogram would hide the individual specification. Points keep their
    exact y; the x offset carries no meaning beyond local density.

    Args:
        ax: target axes.
        values: the values to plot, on the y axis.
        centre: the x position the swarm is built around.
        width: half-width of the swarm, in x data units. The widest row reaches this.
        radius: collision radius in y data units. Defaults to 1.2% of the value range, which
            packs ~127 points into a readable swarm; raise it for fewer, fatter points.
        colour, size, alpha, edgecolor, linewidth, zorder: passed to `ax.scatter`. The thin
            white ring is what keeps two touching points readable as two.

    Returns the x positions used, so a caller can annotate an individual point.

    NOTE: the swarm is normalised to `width` on its own, so swarm *width* is not comparable
    between groups — only the y positions are. Compare heights, never widths.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    x = np.full(v.shape, np.nan)
    if not finite.any():
        return x

    vals = v[finite]
    span = np.ptp(vals) or 1.0
    r = radius if radius is not None else span * 0.012

    # greedy swarm: walk the points in ascending y and give each the smallest |offset| that
    # clears every already-placed point within one diameter vertically
    offsets = np.zeros(len(vals))
    order = np.argsort(vals)
    placed = []
    for i in order:
        y = vals[i]
        blocked = []
        for py, px in placed:
            dy = abs(y - py)
            if dy < 2 * r:
                dx = np.sqrt(max((2 * r) ** 2 - dy ** 2, 0.0))
                blocked.append((px - dx, px + dx))
        candidates = [0.0]
        for lo, hi in blocked:
            candidates += [lo, hi]
        for c in sorted(candidates, key=abs):
            if all(not (lo < c < hi) for lo, hi in blocked):
                offsets[i] = c
                break
        placed.append((y, offsets[i]))

    reach = np.abs(offsets).max()
    offsets = offsets * (width / reach) if reach > 0 else offsets

    x[finite] = centre + offsets
    ax.scatter(x[finite], vals, s=size, color=colour or PALETTE[0], alpha=alpha,
               edgecolor=edgecolor, linewidth=linewidth, zorder=zorder, **kwargs)
    return x


# ====================================================================================================
# continuous colour: the two ramps a choropleth or heatmap may use
# ====================================================================================================
# Before these existed the project had no colormap at all, so every map invented one: notebook 08
# built a seven-stop coral-to-navy ramp, 10_adolescent_analysis built a ten-stop red-to-blue one that
# disagreed with it, and scripts/outcomes.py fell back to raw `viridis`. Three maps of the same
# quantity, three colour languages, none of them the D23 palette that every other figure uses.
#
# Both ramps below are built from `params.colors`, so a change to the palette carries through to the
# maps as well. They interpolate in sRGB, which is adequate for a ramp anchored on two hues chosen
# for their separation; do not add a third hue to the middle of `diverging` — the neutral midpoint is
# what makes the sign of the deviation readable.

def sequential(hue='teal', light='#f7f7f7', name='dgg_sequential'):
    """
    a light-to-hue ramp for a quantity with a natural zero

    Args:
        hue: a key in `params.colors`, or a literal hex string.
        light: the low end. A near-white rather than pure white, so a zero value is still
            distinguishable from the "no data" ground the maps paint underneath.
    """
    from matplotlib.colors import LinearSegmentedColormap

    high = params.colors.get(hue, hue)
    return LinearSegmentedColormap.from_list(name, [light, high])


def diverging(low='rose', high='teal', mid='#f2f0ed', name='dgg_diverging'):
    """
    a two-hue ramp through a neutral midpoint, for a quantity read against a reference value

    For the GGI the reference is parity. Pair this with `matplotlib.colors.TwoSlopeNorm(vcenter=1.0)`
    so the midpoint of the colour scale lands on parity rather than on the midpoint of the data —
    otherwise the colour that reads as "no gap" moves with the sample, which is the failure this
    exists to prevent.

    `rose` -> `teal` is the pairing with the widest separation under simulated protanopia and
    deuteranopia among the D23 hues, which is why it is the default rather than the conventional
    red/blue.
    """
    from matplotlib.colors import LinearSegmentedColormap

    lo = params.colors.get(low, low)
    hi = params.colors.get(high, high)
    return LinearSegmentedColormap.from_list(name, [lo, mid, hi])
