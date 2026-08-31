"""
# Created by valler at 31/08/2026
Feature: the project's colour scheme and figure style — the single source, in one file.

Nothing else in the repo defines a colour. `params.py` re-exports what is here so `params.colors`
and `params.STYLE` keep working; `plotting.py` is the code that applies it. A notebook imports
`plotting as plot` and reads `plot.PALETTE`, `plot.INK`, `plot.EDGE`; it never writes a hex.

    import shared_style as style      # the values
    import plotting as plot          # the functions that apply them

Why a module and not `environment.yml`: the palette is data the analysis reads on every run, so it
has to be importable. `environment.yml` is the conda spec — it pins the interpreter and the
libraries, and nothing in `src/` parses it.
"""

# ====================================================================================================
# the palette (D53, superseding D52)
# ====================================================================================================
# Six steps off ColorBrewer's RdYlBu diverging ramp — one warm end, one cool end, and the pale
# yellow that sits between them. This is a *ramp used as a categorical palette*, which is the
# structure the project's figures actually want: nearly all of them are paired (women against men,
# observed against predicted, gap widening against gap narrowing), and a diverging ramp says
# "opposite ends of one measure" in a way six unrelated hues cannot.
#
# Discriminability, all 15 pairs, worst case (normal / protan / deutan), CIEDE2000:
#
#   worst pair overall        light_blue vs blue        13.3 / 13.1 / 13.4
#   worst same-family pair    light_blue vs blue        13.3 / 13.1 / 13.4
#   worst cross-family pair   cream vs light_blue       34.1 / 34.5 / 39.4
#   best cross-family pair    cream vs navy             77.7 / 80.9 / 82.9
#
# Every pair clears the colour-vision floor of 8, and every cross-family pair clears 34 — the warm
# end and the cool end never collide, under any of the three vision models. What is tight is the
# blue end read against itself: four steps of one hue are separated mostly by lightness, so
# `light_blue`/`blue` (13.1) and `blue`/`steel_blue` (18.5) sit at or below the normal-vision floor
# of 15. That is the cost of taking six steps off one ramp rather than six separate hues, and it is
# why `plot_color_order` fills the well-separated slots first and `plotting.categorical` refuses
# past the sixth rather than cycling.
colors = {
    'red':          '#E66859',   # the warm end#E66859
    'cream':        '#FEE7BA',   # the ramp's midpoint — also what `plotting.diverging` centres on
    'light_blue':   '#75BBD4',
    'blue':         '#74ADD1',
    'steel_blue':   "#416FA0",
    'navy':         "#274668",   # the cool end
}

# The two ends, each light -> dark. Use these when the variable is *paired* (women/men,
# observed/predicted, internet/mobile) rather than a flat list of categories: one family per side of
# the pair, one step per level, which is the reading the scheme is built for. The cool end has four
# steps and the warm end two — that is the ramp's own shape, not an oversight.
families = {
    'warm': ['cream', 'red'],
    'blue': ['light_blue', 'blue', 'steel_blue', 'navy'],
}

# Categorical assignment order. Anchored on the ramp's two ends so a two- or three-series figure
# gets maximum separation (44.8 under the worst vision model), then ordered so each additional
# series costs as little separation as possible. Worst pair among the first k slots:
#
#   k = 2   44.8      k = 4   29.3      k = 6   13.1
#   k = 3   44.8      k = 5   17.6
#
# A figure with four or fewer series is therefore comfortably above the normal-vision floor of 15;
# the fifth and sixth slots are where the blue end starts reading against itself.
plot_color_order = ['red', 'navy', 'light_blue', 'cream', 'steel_blue', 'blue']

# Names the D52 two-family scheme and the older scripts used, mapped onto the nearest surviving step
# so they keep resolving. Nothing active should reach for these — the active call sites were moved
# to the names above. `orange` and `dark_orange` both land on `red`, because the warm end has one
# saturated step and not two: any figure that drew those as two distinct series was updated rather
# than aliased.
colors.update({
    'dark_blue':    colors['navy'],
    'orange':       colors['red'],
    'light_orange': colors['cream'],
    'dark_orange':  colors['red'],
    'teal':         colors['blue'],
    'sky':          colors['light_blue'],
    'purple':       colors['navy'],
    'rose':         colors['red'],
    'magenta':      colors['red'],
    'soft_pink':    colors['red'],
    'peach':        colors['cream'],
    'olive':        colors['cream'],
    'warm_yellow':  colors['cream'],
})

# ====================================================================================================
# ink — everything that is not data
# ====================================================================================================
# Recessive. The data carries the colour; frames, grids and labels stay out of the way.
INK = '#222222'          # titles, and any mark that means "the reference" rather than a series
MUTED = '#666666'        # axis labels, tick labels, reference lines
GRID = '#d9d9d9'         # gridlines and the left/bottom spines

# Mark separator. Every filled mark — bar, wedge, point, patch — is outlined in this, which is what
# keeps two touching marks readable as two. Black, not white: the palette's light steps
# (`light_blue`, `light_orange`) are pale enough that a white outline disappears against the page
# and the marks bleed into each other.
EDGE = '#000000'

# Reserved, outside the categorical order: a mark that says "this value is degenerate", not "this is
# series 4". Used for the zero-floor rings in the GGI validation (D34) and for label highlights.
#
# It was `#C0392B` under the two-family scheme, where the nearest palette hue was 20+ dE2000 away.
# Against the D53 ramp that hex is 4.2 dE from `red` and 1.8 under protanopia — a flag that reads as
# palette slot 0 is not a flag. This dark maroon is the practical ceiling: a ramp spanning
# red -> cream -> navy leaves little colour space outside itself, and nothing clears 17 against both
# the palette and the page. So the flag is not carried by colour alone — the marks that use it are
# rings and outlines, distinguished by shape and weight as well (§6).
FLAG = '#720900'         # 16.5 dE2000 from `red` at worst, 15.2 from INK

# Two more non-hues. Neither is a palette slot: a category drawn in NEUTRAL asserts "this is not one
# of the coloured classes", and NODATA is the ground a map paints where there is no estimate at all.
NEUTRAL = '#9A9A9A'      # a category that is deliberately not a hue — "no material change"
NODATA = '#F2F2F2'       # map ground where nothing was predicted

# ====================================================================================================
# type
# ====================================================================================================
# One family everywhere. The stack is tried left to right and ends in DejaVu Sans, which ships
# inside matplotlib — so a figure still renders, in the right sizes, on a machine that has none of
# the others.
#
# Arial leads, and not Helvetica Neue, for one concrete reason: macOS ships both Helvetica faces as
# `.ttc` collections, and matplotlib reads only the regular face out of a collection. Ask it for
# bold and it silently hands back the regular — so every bold title in this project (`plotting.tidy`
# and `plotting.suptitle` both set `fontweight='bold'`) would lose its emphasis with no error and no
# visible failure but a flat-looking figure. Arial ships Regular/Bold/Italic as separate files, is
# metrically compatible with Helvetica, and is the fallback a Linux or Windows reader would land on
# anyway — so the figures look the same everywhere, and bold means bold.
FONT_STACK = ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans']
MONO_STACK = ['SF Mono', 'Menlo', 'Consolas', 'DejaVu Sans Mono']

# Five steps, and nothing between them. The two smallest exist because the notebooks kept writing
# `tick_fs - 1` and `tick_fs - 2` inline, which is a size decision made in forty places: `annot_fs`
# is the value label on a bar or point, `small_fs` the densest thing a figure is allowed to say
# (a per-country tick in a 30-panel grid). Anything smaller than `small_fs` does not survive being
# scaled into a column, so the ladder stops there.
TITLE_FS = 16            # figure-level title (`plotting.suptitle`)
LABEL_FS = 13            # panel title and axis label
TICK_FS = 11             # tick labels, legend text
ANNOT_FS = 10            # value labels on marks
SMALL_FS = 9             # dense small-multiple labels — the floor

# ====================================================================================================
# figure sizes
# ====================================================================================================
# Named sizes, not literals. Before this, thirty-odd `figsize=(14, 5.4)`-style tuples across the
# notebooks encoded the same handful of intents at slightly different numbers — (13, 6.5), (14, 7)
# and (13, 7) are the same figure drawn three times — so no two panels of the manuscript shared a
# text-to-figure ratio. Pick the name that matches the shape of what is being drawn; the width is
# then constant across the notebooks, which is what makes a font size mean one thing.
#
# Widths cluster at 13 in on purpose: at the ~6.5 in of a two-column page that is a clean 50%
# reduction, so 11 pt tick text lands at 5.5 pt in print.
FIGSIZES = {
    'single':  (8.5, 5.5),    # one panel, one idea
    'square':  (7.8, 7.8),    # 1:1 axes — a scatter read against the identity line
    'wide':    (13, 5.5),     # two panels side by side
    'full':    (13, 7),       # the default: one panel that carries a section
    'tall':    (13, 9),       # 2x2, or one panel with a long category axis
    'banner':  (13, 3.2),     # a single strip — one row of a decomposition
    'poster':  (16, 10.5),    # the big multi-panel grid, its own page
}

# Per-panel extents for a grid whose panel count is data-dependent (one panel per country, one row
# per indicator). `plotting.figsize(nrow=..., ncol=...)` multiplies these out, so a 4-row grid and a
# 9-row grid give their panels the same height instead of squeezing more rows into a fixed box.
ROW_H = 3.3              # height per row of a small-multiple grid
COL_W = 3.4              # width per column of a small-multiple grid
PANEL_H = 4.6            # height per row of FULL-WIDTH panels, which need more room than a grid cell

# `formats` is what `plotting.save` / `plotting.save_fig` write, in order. PDF is first and is the
# one that matters: it is vector, so it survives being scaled into a manuscript. The PNG is the
# convenience copy for a slide or a quick look.
FORMATS = ('pdf', 'png')

STYLE = {
    'colors':   [colors[k] for k in plot_color_order],
    'figsize':  FIGSIZES['full'],
    'figsizes': FIGSIZES,
    'row_h':    ROW_H,
    'col_w':    COL_W,
    'panel_h':  PANEL_H,
    'font':     FONT_STACK,
    'mono':     MONO_STACK,
    'title_fs': TITLE_FS,
    'label_fs': LABEL_FS,
    'tick_fs':  TICK_FS,
    'annot_fs': ANNOT_FS,
    'small_fs': SMALL_FS,
    'dpi':      300,
    'save':     True,           # every figure is written to outputs/fig/<stage>/
    'formats':  FORMATS,
}

# What `plotting` pushes into `matplotlib.rcParams` at import, so the defaults a figure inherits
# when it says nothing are already the project's. Kept here as a plain dict — this module imports
# nothing, which is what lets `params` import it without dragging matplotlib in.
#
# `fonttype: 42` on both vector backends embeds TrueType outlines rather than Type-3, which is what
# a journal's preflight asks for and what keeps the text selectable in the PDF.
RCPARAMS = {
    'font.family':       'sans-serif',
    'font.sans-serif':   FONT_STACK,
    'font.monospace':    MONO_STACK,
    'font.size':         TICK_FS,
    'axes.titlesize':    LABEL_FS,
    'axes.labelsize':    TICK_FS,
    'axes.titleweight':  'bold',
    'axes.titlelocation': 'left',
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   MUTED,
    'xtick.labelsize':   TICK_FS,
    'ytick.labelsize':   TICK_FS,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'legend.fontsize':   TICK_FS,
    'figure.titlesize':  TITLE_FS,
    'figure.figsize':    FIGSIZES['full'],
    'figure.dpi':        110,        # on-screen; `savefig.dpi` is what the written PNG uses
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'grid.color':        GRID,
    'grid.linewidth':    0.6,
    'text.color':        INK,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
}
