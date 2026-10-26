"""
# Created by valler at 29/08/2026
Feature: beta and sigma convergence in the national digital gender gap, estimated so that the
ceiling cannot masquerade as a result.

The analysis this replaces lived entirely inside 08_prediction_and_convergence.ipynb, imported from
dgg_research's `origin/pipeline` branch (D20) and never modernised. Two things were wrong with it and
neither was visible from the output.

**The ceiling made the estimate arithmetic.** Both GGI series are clipped to [0, 1] upstream, and in
2025 97 of 210 internet countries and 126 of 210 mobile ones sit at exactly 1.0. For every one of
them `delta = 1 - initial` holds identically, so the regression of delta on initial returns beta -> -1
with R^2 -> 1 and a vanishing standard error as a matter of arithmetic. South America mobile returned
beta = -1.000, se = 0.000, R^2 = 1.000 and was printed beside genuinely estimated rows with nothing to
distinguish them. D21.5 recorded this and left it uncorrected; D47 is the correction.

**Beta and sigma ran on different samples.** They were computed in separate cells from separately
filtered frames, so a difference between them could not be read. Here `build_sample` is the only
place a sample is constructed and both estimators take its output.

Two further rules the old code broke, both cheap and both load-bearing:

- Standard errors are HC1. Variance shrinks as countries approach the ceiling, so the classical
  homoskedastic SEs understated uncertainty by about 1.7x.
- Every estimate is produced unweighted AND population-weighted, on fixed 2015 adult population, and
  carries which it is. The choice moves beta by about 39% and the sigma fall by 60-70% (D49), so an
  unlabelled convergence figure is less informative than the weighting behind it — the same lesson
  D27 drew for levels.

    import convergence as cv
    panel  = cv.load_panel('internet')
    sample, report = cv.build_sample(panel)
    beta   = cv.beta_convergence(sample, group='continent', weights=cv.fixed_weights())
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import coherent_ggi
import params

CFG = params.CONVERGENCE
INDICATORS = params.COHERENT_GGI['indicators']

# `04_coherent_ggi_figures` writes the annual panel this reads. Globbing for the newest is the
# project's rule for date-stamped artefacts (conventions §4) — never name a dated file here.
PANEL_STAGE = '04_coherent_ggi_figures'
PANEL_PREFIX = 'coherent_ggi_country_year_'

NOTEBOOK = '08_prediction_and_convergence'


# ====================================================================================================
# 1. inputs
# ====================================================================================================
def latest_panel_path():
    """newest `coherent_ggi_country_year_<date>.csv` written by 04"""
    stage = params.table_dir(PANEL_STAGE, create=False)
    files = sorted(stage.glob(f'{PANEL_PREFIX}*.csv'))
    if not files:
        raise FileNotFoundError(
            f'no {PANEL_PREFIX}*.csv in {stage} — run 04_coherent_ggi_figures first')
    return files[-1]


def load_panel(indicator, window=None, path=None):
    """
    country x year panel of the DIRECTLY predicted GGI, complete calendar years only

    Read from 04's annual table rather than by globbing `results/pred_by_year_and_month/`, which is
    what the old notebook did. Same measure — the two agree to a mean absolute difference of 0.0008
    (mobile) and 0.0018 (internet) — but that local copy stops at `2025-07.csv`, so its 2025 was a
    seven-month mean being differenced against a twelve-month 2015. On a rising series that is a
    biased endpoint, not a rounding difference. See D48.
    """
    if indicator not in INDICATORS:
        raise ValueError(f'unknown indicator {indicator!r}; expected one of {INDICATORS}')
    start, end = window or CFG['window']

    frame = pd.read_csv(path or latest_panel_path())
    measure = CFG['measure'].format(ind=indicator)
    keep = ['gid_0', 'country', 'region', 'continent', 'year', measure]
    frame = frame.loc[frame['year'].between(start, end), keep]

    panel = frame.pivot_table(index=['gid_0', 'country', 'region', 'continent'],
                              columns='year', values=measure)
    # A country missing any year of the window cannot contribute to sigma at every point, and its
    # endpoints would rest on a different span than everyone else's.
    return panel.dropna(axis=0, how='any')


def un_subregion():
    """
    UN M49 sub-region per iso3 ('Southern Asia', 'Sub-Saharan Africa', …), as a Series on `gid_0`

    The UN scheme, not the World Bank one the `region` level carries. On the convergence panel the
    two differ by one country: M49 puts Djibouti in Sub-Saharan Africa, the World Bank in the Middle
    East. (M49's Southern Asia also includes Iran, which is not in the panel.) Read from the same
    file `utils.mark_regions` uses.
    """
    m49 = pd.read_csv(params.RAW / 'definition_of_regions.csv', delimiter=';')
    return m49.set_index('ISO-alpha3 Code')['Sub-region Name'].rename_axis('gid_0')


def fixed_weights(year=None, index=None):
    """
    adult population at a single fixed year, as the weight series

    Fixed rather than contemporaneous, for two independent reasons (D49). There is no 2025
    population — `UN_POP_PROCESSED` stops at 2024 and D46 established that its 2024 is a
    byte-identical copy of its 2023 — and fixed weights keep weighted sigma a statement about the
    GGI, since contemporaneous weights make sigma move when population shares move even where no
    country's GGI changes.

    Returns a Series indexed by `gid_0`. Reindexed onto `index` when given, so a caller can align it
    to a sample without a merge.
    """
    year = year or CFG['weight_year']
    pop = coherent_ggi.load_adult_population()
    at_year = pop.loc[pop['year'] == year].set_index('gid_0')['adult_pop']
    if not len(at_year):
        raise ValueError(f'no adult population rows for {year} in {params.UN_POP_PROCESSED}')
    return at_year.reindex(index) if index is not None else at_year


# ====================================================================================================
# 2. the sample — constructed once, shared by both estimators
# ====================================================================================================
def plateau_mask(panel, ceiling=None, tol=None):
    """
    True for countries sitting on the ceiling in EVERY year of the window

    This is the exclusion, and the "every year" is the whole of it. Such a country contributes
    (initial = 1, delta = 0): it is a point mass that adds leverage to the regression and no
    information about convergence, because it never moved.

    A country that CROSSES into the ceiling during the window is deliberately NOT masked. Its
    movement is the finding. D26 refused to drop at-parity countries from the decomposition on
    exactly this ground — dropping them deletes the successes and is regionally selective — and that
    objection does not reach this rule, because a country at the ceiling throughout cannot be a
    country that arrived at it. See D47.
    """
    ceiling = CFG['ceiling'] if ceiling is None else ceiling
    tol = CFG['tol'] if tol is None else tol
    return (panel >= ceiling - tol).all(axis=1)


def build_sample(panel, window=None, exclude_always_at_ceiling=None):
    """
    endpoint frame plus the report of what was dropped to make it

    The single place a sample is constructed. Both `beta_convergence` and `sigma_convergence` take
    what this returns, so the two can never silently run on different countries — which is what the
    notebook this replaces did.

    Returns (sample, report). `sample` carries `initial`, `final`, `delta` and the grouping columns,
    indexed by `gid_0`. `report` is a one-row frame naming the counts, for the record.
    """
    start, end = window or CFG['window']
    exclude = CFG['exclude_always_at_ceiling'] if exclude_always_at_ceiling is None \
        else exclude_always_at_ceiling
    ceiling, tol = CFG['ceiling'], CFG['tol']

    always = plateau_mask(panel)
    at_end = panel[end] >= ceiling - tol
    crossed = at_end & ~always

    kept = panel.loc[~always] if exclude else panel
    sample = pd.DataFrame({'initial': kept[start], 'final': kept[end]},
                          index=kept.index).reset_index()
    sample['delta'] = sample['final'] - sample['initial']
    sample = sample.set_index('gid_0')

    report = pd.DataFrame([{
        'window_start': start, 'window_end': end,
        'n_panel': len(panel),
        'n_always_at_ceiling': int(always.sum()),
        'n_crossed_into_ceiling': int(crossed.sum()),
        'n_at_ceiling_at_end': int(at_end.sum()),
        'n_sample': len(sample),
        'excluded_always_at_ceiling': bool(exclude),
    }])
    return sample, report


# ====================================================================================================
# 3. beta convergence
# ====================================================================================================
def _censored_shares(block, weights, ceiling, tol):
    """what fraction of a group's final values sit on the ceiling, by count and by population"""
    censored = block['final'] >= ceiling - tol
    by_count = float(censored.mean()) if len(block) else np.nan

    by_pop = np.nan
    if weights is not None:
        w = weights.reindex(block.index)
        ok = w.notna()
        if ok.any() and w[ok].sum() > 0:
            by_pop = float(w[ok & censored].sum() / w[ok].sum())
    return by_count, by_pop


def _fit_one(block, label, weights, cov_type, ceiling, tol, span):
    """one beta regression, or a row explaining why there is not one"""
    by_count, by_pop = _censored_shares(block, weights, ceiling, tol)
    row = {'group': label, 'n': len(block),
           'censored_share_end': by_count, 'censored_share_end_popwt': by_pop,
           'weighting': 'pop_weighted' if weights is not None else 'unweighted'}

    w = None
    if weights is not None:
        w = weights.reindex(block.index)
        block = block.loc[w.notna()]
        w = w.dropna()
        row['n'] = len(block)

    # A group needs more countries than parameters and some spread in the initial level; a group
    # that is entirely at one value has no regression to run, only an average.
    enough = len(block) >= 5 and np.ptp(block['initial'].values) > tol
    if not enough:
        return {**row, 'beta': np.nan, 'se': np.nan, 'r2': np.nan, 'pval': np.nan,
                'lambda': np.nan, 'identified': False,
                'note': 'too few countries or no spread in the initial level'}

    X = sm.add_constant(block['initial'])
    fit = (sm.WLS(block['delta'], X, weights=w.values) if w is not None
           else sm.OLS(block['delta'], X)).fit(cov_type=cov_type)
    beta = float(fit.params['initial'])

    # The implied annual speed of convergence. Undefined once beta <= -1, which is exactly the
    # saturated case: a group where every country lands on the ceiling has closed its entire gap in
    # one step and no rate describes it.
    lam = float(-np.log1p(beta) / span) if beta > -1 else np.nan

    share = by_pop if (weights is not None and not np.isnan(by_pop)) else by_count
    identified = bool(share <= CFG['max_censored_share'])
    note = '' if identified else (
        f'{share:.0%} of the group is on the ceiling at the endpoint — delta = 1 - initial holds '
        f'for most of it, so beta is arithmetic rather than evidence')

    return {**row, 'beta': beta, 'se': float(fit.bse['initial']), 'r2': float(fit.rsquared),
            'pval': float(fit.pvalues['initial']), 'lambda': lam,
            'identified': identified, 'note': note}


def beta_convergence(sample, group=None, weights=None, cov_type=None, window=None):
    """
    cross-sectional regression of the change on the initial level, globally and per group

        (GGI_end - GGI_start)_i = a + beta * GGI_start,i + e_i

    A negative beta is the convergence signal. Three things about the estimate are not optional:

    **`identified` is part of the result.** Where most of a group ends on the ceiling, `delta` is
    `1 - initial` by construction and beta tends to -1 with R^2 -> 1 whatever the countries did. Such
    a group is returned with `identified = False` and a note saying so, rather than dropped — a
    reader who sees no row assumes an error, a reader who sees a flagged row learns something.

    **`lambda` is the comparable number.** `beta` is the total effect over the whole window, so it is
    not comparable to any published convergence rate without dividing through: `lambda =
    -ln(1 + beta) / span`.

    **`beta < 0` does not establish convergence.** `initial` appears on both sides of the regression,
    so any noise in it biases beta negative — the standard regression-to-the-mean objection. Read
    beta together with `sigma_convergence`, which is the test that dispersion actually fell.

    Args:
        sample: the frame from `build_sample`.
        group: a column of `sample` to estimate within, in addition to the global row.
        weights: a Series indexed like `sample` (from `fixed_weights`) for WLS, or None for OLS.
        cov_type: passed to statsmodels; HC1 by default because variance shrinks near the ceiling.
    """
    cov_type = cov_type or CFG['cov_type']
    start, end = window or CFG['window']
    span = end - start
    ceiling, tol = CFG['ceiling'], CFG['tol']

    rows = [_fit_one(sample, 'Global', weights, cov_type, ceiling, tol, span)]
    if group:
        for label, block in sample.groupby(group):
            rows.append(_fit_one(block, label, weights, cov_type, ceiling, tol, span))

    out = pd.DataFrame(rows)
    return out[['group', 'weighting', 'n', 'beta', 'se', 'r2', 'pval', 'lambda',
                'censored_share_end', 'censored_share_end_popwt', 'identified', 'note']]


# ====================================================================================================
# 4. sigma convergence
# ====================================================================================================
def _weighted_std(values, w):
    """population-weighted standard deviation, the frequency-weight form"""
    mean = np.average(values, weights=w)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=w)))


def _sigma_by_year(block, years, w=None):
    """the dispersion series for one set of countries"""
    return [(_weighted_std(block[y].values, w) if w is not None
             else float(np.std(block[y].values, ddof=1))) for y in years]


def _bootstrap_change(block, years, w, draws, rng):
    """
    percentile CI for the first-to-last change in sigma, resampling COUNTRIES

    The eleven yearly sigmas are computed from the same countries, so they are not independent
    observations and an OLS p-value on a regression of sigma against year does not test what it
    appears to. The sampling variation that matters is over which countries are in the group, which
    is what this resamples. `slope_per_year` is kept as a descriptive summary of the path; this is
    the interval to quote.
    """
    n = len(block)
    if n < 5:
        return np.nan, np.nan
    changes = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, n, n)
        sub = block.iloc[pick]
        wb = None if w is None else w[pick]
        s = _sigma_by_year(sub, [years[0], years[-1]], wb)
        changes[d] = s[-1] - s[0]
    return float(np.percentile(changes, 2.5)), float(np.percentile(changes, 97.5))


def sigma_convergence(panel, sample=None, group=None, weights=None, window=None,
                      draws=1000, seed=None):
    """
    cross-country dispersion by year, and a test that it actually falls

    Sigma convergence is the substantive claim. Beta convergence is necessary for it but not
    sufficient, because a negative beta arises from regression to the mean alone whenever the initial
    level is measured with any noise — and `initial` sits on both sides of that regression. The
    notebook this replaces plotted sigma and never tested it, so "dispersion is falling" rested on
    reading a line off a chart.

    Restricted to `sample`'s countries when one is given, so sigma and beta describe the same set.

    Returns (series, trend). `series` is sigma per group-year. `trend` carries the first-to-last
    change with a **bootstrap CI over countries** (`ci_low`, `ci_high`) — the interval to quote —
    alongside a descriptive `slope_per_year`. `converging` is set from the CI, not from the slope's
    p-value; see `_bootstrap_change` for why the latter would be misleading.
    """
    start, end = window or CFG['window']
    years = [y for y in panel.columns if start <= y <= end]
    rng = np.random.default_rng(params.SEED if seed is None else seed)

    frame = panel
    if sample is not None:
        frame = panel[panel.index.get_level_values('gid_0').isin(sample.index)]

    w_all = None
    if weights is not None:
        w_all = np.asarray(weights.reindex(frame.index.get_level_values('gid_0')), dtype=float)

    weighting = 'pop_weighted' if weights is not None else 'unweighted'
    # Global first, then each group — the same shape `beta_convergence` returns.
    blocks = [('Global', np.ones(len(frame), dtype=bool))]
    if group:
        keys = frame.index.get_level_values(group)
        blocks += [(label, np.asarray(keys == label)) for label in pd.unique(keys)]

    rows, trend = [], []
    for label, mask in blocks:
        block = frame.loc[mask]
        wb = None if w_all is None else w_all[mask]
        if wb is not None:
            ok = ~np.isnan(wb)
            block, wb = block.loc[ok], wb[ok]
        if len(block) < 2:
            continue

        sigmas = _sigma_by_year(block, years, wb)
        for year, sigma in zip(years, sigmas):
            rows.append({'group': label, 'year': year, 'n': len(block),
                         'weighting': weighting, 'sigma': sigma})

        fit = sm.OLS(sigmas, sm.add_constant(np.asarray(years, dtype=float))).fit(
            cov_type=CFG['cov_type'])
        lo, hi = _bootstrap_change(block, years, wb, draws, rng)
        change = sigmas[-1] - sigmas[0]
        trend.append({'group': label, 'weighting': weighting, 'n_countries': len(block),
                      'sigma_start': sigmas[0], 'sigma_end': sigmas[-1], 'sigma_change': change,
                      'ci_low': lo, 'ci_high': hi,
                      'slope_per_year': float(fit.params[1]), 'r2': float(fit.rsquared),
                      'converging': bool(change < 0 and not np.isnan(hi) and hi < 0)})

    return pd.DataFrame(rows), pd.DataFrame(trend)


# ====================================================================================================
# 5. the run
# ====================================================================================================
def run(indicator, group=None, window=None):
    """
    every convergence table for one indicator, both weightings

    The notebook calls this and plots what comes back; nothing here needs a notebook to run, which
    is the point (D38).
    """
    group = group or CFG['group']
    panel = load_panel(indicator, window=window)
    sample, report = build_sample(panel, window=window)
    report.insert(0, 'indicator', indicator)

    w = fixed_weights(index=sample.index)

    betas, sigmas, trends = [], [], []
    for weights in (None, w):
        betas.append(beta_convergence(sample, group=group, weights=weights, window=window))
        s, t = sigma_convergence(panel, sample=sample, group=group, weights=weights, window=window)
        sigmas.append(s)
        trends.append(t)

    out = {'sample_report': report,
           'beta': pd.concat(betas, ignore_index=True),
           'sigma': pd.concat(sigmas, ignore_index=True),
           'sigma_trend': pd.concat(trends, ignore_index=True)}
    for frame in out.values():
        if 'indicator' not in frame.columns:
            frame.insert(0, 'indicator', indicator)
    return out


def beta_latex(beta, weighting='unweighted', caption=None, label=None, decimals=3):
    """
    the beta-convergence table as a LaTeX `table`, in the manuscript's booktabs layout

    Deliberately carries no significance stars. Every row in this table is p < 0.001, so a star
    column separates nothing; worse, the tightest standard errors belong to the groups that are
    almost entirely censored at the parity ceiling, where delta = 1 - initial holds identically and
    beta is arithmetic rather than evidence (D47). Stars there would read as the strongest result on
    the page. The `ceiling` column and the dagger carry that instead.

    One weighting per table, named in the caption, because beta is not comparable across the two
    (D49). `lambda` is left out: on a fully censored group it is not interpretable — mobile South
    America returns 3.21/yr — and the two global rows are quoted in the notes instead.

    Needs `booktabs` in the preamble and nothing else.
    """
    names = {'unweighted': 'unweighted', 'pop_weighted': 'population-weighted'}
    if weighting not in names:
        raise ValueError(f'weighting must be one of {sorted(names)}, got {weighting!r}')
    # the censoring share the identification test used for THIS weighting, so the dagger always
    # agrees with the number beside it (internet Oceania is 30% by country and 76% by population)
    share_col = ('censored_share_end' if weighting == 'unweighted'
                 else 'censored_share_end_popwt')
    unit = 'countries' if weighting == 'unweighted' else 'population'

    block = beta[beta['weighting'] == weighting]
    indicators = [i for i in INDICATORS if i in set(block['indicator'])]
    groups = list(dict.fromkeys(block['group']))
    keyed = block.set_index(['indicator', 'group'])

    start, end = CFG['window']
    caption = caption or (f'Results of $\\beta$-convergence analysis by continent, {start}--{end} '
                          f'({names[weighting]})')
    label = label or f'tab:beta_convergence_{weighting}'

    lines = [r'\begin{table}[!htbp]', r'\centering', f'\\caption{{{caption}}}',
             f'\\label{{{label}}}',
             r'\begin{tabular}{l' + 'rcr' * len(indicators) + '}', r'\toprule']
    lines.append(' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{i.capitalize()}}}'
                                    for i in indicators) + r' \\')
    lines.append(' '.join(f'\\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}'
                          for k in range(len(indicators))))
    lines.append('Analysis Level & ' + ' & '.join(
        [r'$n$ & $\beta$ (HC1 s.e.) & ceiling'] * len(indicators)) + r' \\')
    lines.append(r'\midrule')

    for group in groups:
        cells = []
        for indicator in indicators:
            if (indicator, group) not in keyed.index:
                cells += ['', '', '']
                continue
            r = keyed.loc[(indicator, group)]
            # the dagger goes on the ESTIMATE, not on the row name: identification is per
            # indicator, and Oceania is the case that proves it — internet Oceania is identified
            # unweighted (30% of countries on the ceiling) while mobile Oceania is not (53%).
            # A row-level flag would claim both were censored, or neither.
            flag = '' if bool(r['identified']) else r'$^{\dagger}$'
            cells += [f'{int(r["n"])}',
                      f'{r["beta"]:.{decimals}f} ({r["se"]:.{decimals}f}){flag}',
                      f'{100 * r[share_col]:.0f}\\%']
        lines.append(f'{group} & ' + ' & '.join(cells) + r' \\')
        if group == 'Global':
            lines.append(r'\addlinespace')

    lam = {i: keyed.loc[(i, 'Global'), 'lambda'] for i in indicators}
    lam_text = ', '.join(f'{i} {lam[i]:.3f}/yr' for i in indicators)
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\begin{flushleft}\footnotesize',
              rf'\textit{{Notes:}} Dependent variable is the change in the directly predicted '
              rf'gender gap index over {start}--{end}, regressed on its {start} level; '
              rf'HC1 standard errors in parentheses. Estimates are {names[weighting]}. '
              rf'\emph{{ceiling}} is the share of the group ({unit}) sitting at parity at the '
              rf'endpoint.\\',
              r'$^{\dagger}$ Not identified: the group is mostly censored at the parity ceiling, '
              r'where $\Delta = 1 - \text{initial}$ holds by construction, so $\beta$ there is '
              r'arithmetic rather than evidence and is reported for completeness only.\\',
              rf'Implied annual speed of convergence at the global level: {lam_text}. '
              rf'No significance stars are shown: every coefficient is $p<0.001$, and the smallest '
              rf'standard errors belong to the least identified rows.',
              r'\end{flushleft}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def write(frames, stamp=None):
    """date-stamped tables into outputs/tables/08_prediction_and_convergence/ (conventions §4)"""
    import datetime

    stamp = stamp or datetime.date.today().strftime('%Y%m%d')
    outdir = params.table_dir(NOTEBOOK)
    written = []
    for name, frame in frames.items():
        path = outdir / f'convergence_{name}_{stamp}.csv'
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def main():
    frames = {}
    for indicator in INDICATORS:
        for name, frame in run(indicator).items():
            frames.setdefault(name, []).append(frame)
    merged = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    for path in write(merged):
        print(f'wrote {path}')
    return merged


if __name__ == '__main__':
    main()
