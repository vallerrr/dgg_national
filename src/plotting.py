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
import shared_style as style

# Everything below is re-exported from shared_style so a notebook can reach it through `plot.`
# without a second import. shared_style.py is where the values are edited (D52).
STYLE = params.STYLE
PALETTE = STYLE['colors']
FAMILIES = style.families

INK, MUTED, GRID = style.INK, style.MUTED, style.GRID
EDGE = style.EDGE            # mark separator — black, see shared_style
FLAG = style.FLAG            # "this value is degenerate", not "this is series 4" (D34)
NEUTRAL = style.NEUTRAL      # a category that is deliberately not a hue
NODATA = style.NODATA        # map ground where nothing was predicted

FIGSIZES = style.FIGSIZES    # named figure sizes — see `figsize()` below
FONT_STACK = style.FONT_STACK


def use_style():
    """
    push `shared_style.RCPARAMS` into matplotlib's defaults

    Called once on import, so a notebook that does nothing but `import plotting as plot` already
    draws in the project's font, sizes and ink. `tidy()` still sets the same things on the axes it
    is given — it has to, because a figure built before this ran, or one whose axes were restyled
    by a library, would otherwise keep matplotlib's defaults. The two agree by construction: both
    read `shared_style`.
    """
    plt.rcParams.update(style.RCPARAMS)
    return plt.rcParams


use_style()


def figsize(name='full', nrow=None, ncol=None, scale=1.0, row_h=None, col_w=None):
    """
    a figure size from the project's named set, or one derived from a panel grid

    Args:
        name: a key of `shared_style.FIGSIZES` — 'single', 'square', 'wide', 'full', 'tall',
            'banner', 'poster'. Ignored on an axis that `nrow`/`ncol` supplies.
        nrow: number of panel rows. Given, the height becomes `nrow * STYLE['row_h']` — so every
            panel of a data-dependent grid gets the same height instead of the grid being squeezed
            into a fixed box as countries are added.
        ncol: number of panel columns; sets the width the same way.
        scale: multiplies both extents. For the one figure that needs to be half again as big
            without inventing a new name.
        row_h, col_w: override the per-panel extents. The one real use is a stack of full-width
            panels, which needs more height per row than a grid cell does — pass
            `row_h=STYLE['panel_h']` for that rather than writing a number.

    Returns (width, height) in inches.

        plt.subplots(1, 2, figsize=plot.figsize('wide'))
        plt.subplots(nrow, 3, figsize=plot.figsize('full', nrow=nrow))   # fixed width, grown height
        plt.subplots(nrow, ncol, figsize=plot.figsize(nrow=nrow, ncol=ncol))
    """
    try:
        w, h = FIGSIZES[name]
    except KeyError:
        raise KeyError(f'{name!r} is not a project figure size; pick one of '
                       f'{", ".join(FIGSIZES)}') from None
    if ncol is not None:
        w = ncol * (col_w or STYLE['col_w'])
    if nrow is not None:
        h = nrow * (row_h or STYLE['row_h'])
    return (w * scale, h * scale)


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
                      markerfacecolor=colour, markeredgecolor=EDGE, label=label)


def categorical(categories=None, order=None, extend=False):
    """
    map categories onto the project palette, deterministically

    Args:
        categories: the category values to colour. Sorted before assignment, so a category keeps
            the same hue in every figure that draws it. Omit when passing `order`.
        order: an explicit category order, for when sorted order is not the meaningful one.

    Returns {category: hex}.

    Zipping PALETTE against `Series.unique()` is the trap this closes: `unique()` returns *data*
    order, so the same region takes a different hue in each figure of the same notebook, and a
    reader who carries a colour across two panels is misled by it.

        extend: allow more categories than the palette has slots. Off by default, because the
            honest answer to a seventh category is usually to fold the tail into "Other", use small
            multiples, or a highlight-plus-context treatment. When on, the extra colours are read
            off the two family ramps (see `spread`) — still one scheme, but the pairwise separation
            the six checked slots guarantee no longer holds.
    """
    if order is None and categories is None:
        raise TypeError('pass categories, or an explicit order')
    cats = list(order) if order is not None else sorted(
        {c for c in categories if c is not None and c == c})
    if len(cats) > len(PALETTE):
        if not extend:
            raise ValueError(
                f'{len(cats)} categories against {len(PALETTE)} hues (D52). The scheme is two hue '
                'families, so six is the cap: fold the tail into "Other", use small multiples, or '
                'pass extend=True and accept weaker separation.')
        return dict(zip(cats, spread(len(cats))))
    return dict(zip(cats, PALETTE))


def spread(n):
    """
    `n` colours read off the two family ramps, alternating blue and orange

    The overflow path for `categorical(extend=True)`, and the right choice whenever a variable has
    more levels than the palette has slots but still wants to look like this scheme. Colours stay
    on the two ends of the ramp; what is lost is the checked pairwise separation.

    Each family is interpolated along its own steps rather than between a light and a dark anchor,
    so the cool end's four steps all contribute — walking `light_blue -> navy` in a straight line
    would slide past `blue` and `steel_blue` and produce colours the scheme does not contain.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    per = int(np.ceil(n / len(FAMILIES)))
    ramps = []
    for fam, steps in FAMILIES.items():
        cmap = LinearSegmentedColormap.from_list(f'dgg_{fam}', [style.colors[s] for s in steps])
        ramps.append([to_hex(cmap(x)) for x in np.linspace(0, 1, per)])
    out = [c for pair in zip(*ramps) for c in pair]
    return out[:n]


def save_fig(fig, path, dpi=None):
    """
    write one figure in every format in STYLE['formats'] — PDF first, then PNG

    Args:
        fig: the figure to write.
        path: the target path. Its suffix is *replaced* per format, so callers can keep passing the
            `.png` path they always passed and still get the PDF.
        dpi: overrides STYLE['dpi']; only affects the raster formats.

    Returns the list of paths written, or [] when STYLE['save'] is off.

    PDF is the one that matters — it is vector, so a figure scaled into a manuscript stays sharp.
    The PNG is the convenience copy.
    """
    from pathlib import Path

    if not STYLE['save']:
        return []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in STYLE['formats']:
        target = path.with_suffix(f'.{ext}')
        fig.savefig(target, dpi=dpi or STYLE['dpi'], bbox_inches='tight')
        written.append(target)
    return written


def save(fig, stage, name):
    """
    write a figure to `outputs/fig/<stage>/<name>.{pdf,png}` when STYLE['save'] is on

    The stage folder is the same one the producing notebook sits in, so a figure's path names the
    step that made it. Pass `name` without a suffix.
    """
    return save_fig(fig, params.fig_dir(stage) / f'{name}.png')


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
        ax.bar(pos, vals, width * 0.88, color=colours[name], edgecolor=EDGE,
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
                        fontsize=STYLE['annot_fs'], color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    return ax


def beeswarm(ax, values, centre=0.0, width=0.34, radius=None, colour=None, size=22,
             alpha=0.75, edgecolor=EDGE, linewidth=0.3, zorder=3, **kwargs):
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
            outline is what keeps two touching points readable as two.

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
# quantity, three colour languages, none of them the project palette every other figure uses.
#
# Both ramps below are built from `shared_style.colors`, so a change to the palette carries through
# to the maps as well. They interpolate in sRGB, which is adequate for a ramp anchored on two hues
# chosen for their separation; do not add a third hue to the middle of `diverging` — the neutral
# midpoint is what makes the sign of the deviation readable.

def sequential(hue='navy', light='#F7F7F7', name='dgg_sequential'):
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


def diverging(low='red', high='navy', mid='cream', name='dgg_diverging'):
    """
    a two-hue ramp through a neutral midpoint, for a quantity read against a reference value

    For the GGI the reference is parity. Pair this with `matplotlib.colors.TwoSlopeNorm(vcenter=1.0)`
    so the midpoint of the colour scale lands on parity rather than on the midpoint of the data —
    otherwise the colour that reads as "no gap" moves with the sample, which is the failure this
    exists to prevent.

    `red` -> `cream` -> `navy` is the scheme itself: the palette is six steps off RdYlBu, so a
    diverging map built from its two ends and its own midpoint is the ramp put back together (D53).
    It is also the classic colour-vision-safe diverging pairing — the two ends stay 45 dE2000 apart
    under simulated protanopia and 67 under deuteranopia, so the sign of the deviation survives
    where a red/green ramp would not.
    """
    from matplotlib.colors import LinearSegmentedColormap

    lo = params.colors.get(low, low)
    hi = params.colors.get(high, high)
    md = params.colors.get(mid, mid)
    return LinearSegmentedColormap.from_list(name, [lo, md, hi])


def ordinal(n, hue='navy', lo=0.30, hi=1.0):
    """
    `n` colours stepping along `sequential(hue)`, for a category that is ordered

    HDI bands, income quartiles, year bins: the categories have a direction, and a handful of
    unrelated hues assert they do not. A ramp says "these are ranked" with no extra ink. Starts at `lo`
    rather than 0 so the first band is not near-white.
    """
    from matplotlib.colors import to_hex

    cmap = sequential(hue)
    return [to_hex(cmap(x)) for x in np.linspace(lo, hi, n)]
