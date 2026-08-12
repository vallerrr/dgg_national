"""
# Created by valler at 05/08/2026
Feature: exact decomposition of coherent-GGI change into female and male adoption change.

Because the coherent GGI is a ratio, its log change splits *exactly* into a female and a male term:

    GGI_ct                = female_ct / male_ct
    delta_log_GGI         = log(GGI_c,t1)    - log(GGI_c,t0)
    female_component      = log(female_c,t1) - log(female_c,t0)
    male_component        = -[log(male_c,t1) - log(male_c,t0)]
    delta_log_GGI         = female_component + male_component      (identity, verified numerically)

The identity is what makes the exercise worth doing: a rising GGI can come from women gaining access
or from men losing it, and those are not the same finding. Every output therefore carries the
direction of **parity** and the direction of **female access** as separate fields, and the typology
never labels a male-decline convergence as an access gain.

Zeros are excluded rather than patched: log(0) is undefined and no constant is added to the primary
results. A separate epsilon sensitivity is reported so the cost of that choice is visible.

    python src/15_coherent_ggi_decomposition.py
"""
from datetime import datetime

import numpy as np
import pandas as pd

import params

CFG = params.COHERENT_GGI
INDICATORS = CFG['indicators']
OUTDIR = params.OUTPUTS / 'tables'
PREFIX = 'ggi_decomposition'

TREND_START, TREND_END = 2015, 2025      # 2026 is partial (D18)
PRIMARY = (2015, 2025)
EXTRA_INTERVALS = [(2015, 2020), (2020, 2025)]

TAU = 0.01           # log-point threshold for "meaningful"; ~1% relative change
TOLERANCE = 1e-12    # the decomposition identity is exact, so the tolerance is machine-level
EPSILON = 0.001      # sensitivity only — never used in the primary results
WEIGHT_YEAR = 2015   # fixed base-year population weights (see `load_population`)


# ====================================================================================================
# codebook
# ====================================================================================================
CODEBOOK = [
    dict(code=1, label='Female-led expansion',
         parity='improving', female_access='expanding',
         rule='delta > tau; f > tau; m <= tau',
         reading='Women gained access and the gap narrowed; men flat or declining. Both a parity '
                 'improvement and an access expansion for women.'),
    dict(code=2, label='Shared expansion with convergence',
         parity='improving', female_access='expanding',
         rule='delta > tau; f > tau; m > tau',
         reading='Both sexes gained access, women proportionally faster. Parity improvement '
                 'accompanied by access expansion for both.'),
    dict(code=3, label='Convergence through male decline',
         parity='improving', female_access='stable',
         rule='delta > tau; |f| <= tau; m < -tau',
         reading='The gap narrowed because male adoption fell, not because women gained access. '
                 'A parity improvement WITHOUT access expansion — must not be reported as improved '
                 'digital access.'),
    dict(code=4, label='Shared contraction with convergence',
         parity='improving', female_access='contracting',
         rule='delta > tau; f < -tau; m < -tau',
         reading='Both sexes lost access, men faster. A parity improvement during access '
                 'contraction — must not be reported as improved digital access.'),
    dict(code=5, label='Divergence during expansion',
         parity='worsening', female_access='expanding or stable',
         rule='delta < -tau; m > tau; f > -tau',
         reading='Access grew but men gained proportionally faster, so the gap widened.'),
    dict(code=6, label='Female-led contraction and divergence',
         parity='worsening', female_access='contracting',
         rule='delta < -tau; f < -tau',
         reading='Female adoption fell and the gap widened.'),
    dict(code=8, label='At or above parity throughout',
         parity='at parity', female_access='varies',
         rule='female >= male at BOTH endpoints (requires level_ceiling_ctrl)',
         reading='Women\'s predicted adoption met or exceeded men\'s at both endpoints, so there '
                 'was no gap against women to close. Reported as a standing state, not as change. '
                 'A country that crosses INTO parity during the window is not in this category — '
                 'its convergence is the finding and it keeps its convergence category.'),
    dict(code=7, label='Ambiguous or negligible change',
         parity='stable', female_access='varies',
         rule='|delta| <= tau, or a sign pattern not covered above',
         reading=f'Parity change below the {TAU} log-point threshold, or a combination of level '
                 'changes with no clear reading.'),
]

# categories where parity improves but women did not gain access
PARITY_WITHOUT_ACCESS = {3, 4}


def classify(f, m):
    """
    assign one typology code to a country-period

    Mutually exclusive and exhaustive by construction: the branches are evaluated in order and the
    final `else` catches every remaining sign pattern. `f` and `m` are log changes in female and
    male adoption; `delta = f - m` is the log change in the GGI.
    """
    if not np.isfinite(f) or not np.isfinite(m):
        return np.nan
    delta = f - m

    if abs(delta) <= TAU:
        return 7
    if delta > 0:                                  # parity improving
        if f > TAU:
            return 2 if m > TAU else 1
        if f < -TAU:
            return 4 if m < -TAU else 7
        return 3 if m < -TAU else 7                # female flat: only a male decline can explain it
    # parity worsening
    if f < -TAU:
        return 6
    if m > TAU:
        return 5
    return 7


# ====================================================================================================
# inputs
# ====================================================================================================
def latest(stem):
    matches = sorted(OUTDIR.glob(f'{stem}_*.csv'))
    if not matches:
        raise FileNotFoundError(f'no {stem}_*.csv — run src/13_coherent_ggi.py first')
    return pd.read_csv(matches[-1])


def load_levels():
    """annual female and male adoption levels per country"""
    annual = latest('coherent_ggi_country_year')
    return annual[annual['year'].between(TREND_START, TREND_END)].copy()


def load_population():
    """
    sex-specific adult population, fixed at the base year

    Weights are held at `WEIGHT_YEAR` rather than moving with time for two reasons: the UN file
    stops at 2024 so 2025 has no population of its own (doc/data_updates.md), and fixed weights keep
    the regional decomposition exactly additive — with time-varying weights a regional change mixes
    adoption change with population composition change, which is not what this decomposition claims
    to measure.
    """
    pop = pd.read_csv(params.RAW / 'un_1950_2023_processed.csv',
                      usecols=['iso3', 'Year', '18_inf_m', '18_inf_f'])
    pop = pop[pop['Year'] == WEIGHT_YEAR]
    return pop.rename(columns={'iso3': 'gid_0', '18_inf_f': 'pop_female', '18_inf_m': 'pop_male'})[
        ['gid_0', 'pop_female', 'pop_male']]


# ====================================================================================================
# decomposition
# ====================================================================================================
def apply_level_ceiling(df):
    """
    cap the female level at the male level, so parity is the ceiling

    Equivalent to capping the GGI at 1, but applied to the level so the ratio stays a ratio and the
    decomposition identity survives. Where the cap binds, the female component measures the change
    in min(female, male) rather than in female adoption alone — flagged per row so the distinction
    stays visible.
    """
    df['at_parity_t0'] = df['f0'] >= df['m0']
    df['at_parity_t1'] = df['f1'] >= df['m1']
    # at parity at *both* endpoints — the country was already there and stayed there. A country that
    # crosses into parity during the window is not this: its movement is the finding, and it keeps
    # whatever convergence category it earned.
    df['at_parity_both'] = df['at_parity_t0'] & df['at_parity_t1']
    df['ceiling_binding'] = df['at_parity_t0'] | df['at_parity_t1']

    df['f0'] = np.minimum(df['f0'], df['m0'])
    df['f1'] = np.minimum(df['f1'], df['m1'])
    return df


def decompose(levels, indicator, t0, t1, epsilon=0.0, ceiling=None):
    """
    log decomposition for one indicator over one interval

    With `epsilon = 0` (the default, used for every primary result) a country is dropped whenever
    either level is zero or missing at either endpoint, because the log change does not exist.
    """
    ceiling = params.level_ceiling_ctrl if ceiling is None else ceiling
    female, male = f'{indicator}_women', f'{indicator}_men'
    keys = ['gid_0', 'country', 'region', 'continent']

    start = levels[levels['year'] == t0].set_index('gid_0')
    end = levels[levels['year'] == t1].set_index('gid_0')
    common = start.index.intersection(end.index)

    df = start.loc[common, keys[1:]].copy()
    df.index.name = 'gid_0'
    for name, src, yr in [('f0', start, t0), ('f1', end, t1)]:
        df[name] = src.loc[common, female]
    for name, src in [('m0', start), ('m1', end)]:
        df[name] = src.loc[common, male]

    df = (apply_level_ceiling(df) if ceiling else
          df.assign(at_parity_t0=False, at_parity_t1=False,
                    at_parity_both=False, ceiling_binding=False))

    if epsilon:
        for c in ['f0', 'f1', 'm0', 'm1']:
            df[c] = df[c].where(df[c] > 0, epsilon)

    usable = (df[['f0', 'f1', 'm0', 'm1']] > 0).all(axis=1) & df[['f0', 'f1', 'm0', 'm1']].notna().all(axis=1)

    out = df[usable].copy()
    out['female_component'] = np.log(out['f1']) - np.log(out['f0'])
    out['male_component'] = -(np.log(out['m1']) - np.log(out['m0']))
    out['delta_log_ggi'] = (np.log(out['f1'] / out['m1'])) - (np.log(out['f0'] / out['m0']))
    out['identity_residual'] = (out['delta_log_ggi']
                                - (out['female_component'] + out['male_component']))

    out['male_log_change'] = np.log(out['m1']) - np.log(out['m0'])
    out['category'] = [classify(f, m) for f, m in zip(out['female_component'], out['male_log_change'])]

    # Standing parity is a state, not a change. With the ceiling on, these rows already classify as
    # 7 (both capped GGIs equal 1, so delta is exactly 0); code 8 separates them from genuine
    # below-threshold movement without touching any other category.
    if ceiling:
        out.loc[out['at_parity_both'], 'category'] = 8

    labels = {c['code']: c['label'] for c in CODEBOOK}
    out['category_label'] = out['category'].map(labels)
    out['parity_direction'] = np.where(out['delta_log_ggi'] > TAU, 'improving',
                                np.where(out['delta_log_ggi'] < -TAU, 'worsening', 'stable'))
    out['female_access_direction'] = np.where(out['female_component'] > TAU, 'expanding',
                                       np.where(out['female_component'] < -TAU, 'contracting', 'stable'))
    out['parity_improved_without_access_gain'] = out['category'].isin(PARITY_WITHOUT_ACCESS)
    out['level_ceiling_applied'] = bool(ceiling)

    out = out.reset_index()
    out.insert(0, 'indicator', indicator)
    out.insert(1, 'interval', f'{t0}-{t1}')
    out.insert(2, 't0', t0)
    out.insert(3, 't1', t1)

    excluded = df[~usable]
    exclusion = {
        'indicator': indicator, 'interval': f'{t0}-{t1}', 'epsilon': epsilon,
        'n_countries_available': len(df),
        'n_included': int(usable.sum()),
        'n_excluded': int((~usable).sum()),
        'n_excluded_zero_female': int(((excluded[['f0', 'f1']] <= 0).any(axis=1)).sum()),
        'n_excluded_zero_male': int(((excluded[['m0', 'm1']] <= 0).any(axis=1)).sum()),
        'n_excluded_missing': int(excluded[['f0', 'f1', 'm0', 'm1']].isna().any(axis=1).sum()),
        'excluded_iso3': ','.join(sorted(excluded.index)) if len(excluded) else '',
    }
    return out, exclusion


def build_all(levels):
    """primary interval, the two sub-periods, and every consecutive annual pair"""
    intervals = [PRIMARY] + EXTRA_INTERVALS + [(y, y + 1) for y in range(TREND_START, TREND_END)]
    rows, exclusions = [], []
    for indicator in INDICATORS:
        for t0, t1 in intervals:
            out, exc = decompose(levels, indicator, t0, t1)
            rows.append(out)
            exclusions.append(exc)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(exclusions)


def sensitivity(levels):
    """what the epsilon substitution would change, reported rather than adopted"""
    rows = []
    for indicator in INDICATORS:
        for t0, t1 in [PRIMARY] + EXTRA_INTERVALS:
            base, base_exc = decompose(levels, indicator, t0, t1, epsilon=0.0)
            eps, eps_exc = decompose(levels, indicator, t0, t1, epsilon=EPSILON)
            shared = base.set_index('gid_0').index.intersection(eps.set_index('gid_0').index)
            b = base.set_index('gid_0').loc[shared, 'delta_log_ggi']
            e = eps.set_index('gid_0').loc[shared, 'delta_log_ggi']
            rows.append({
                'indicator': indicator, 'interval': f'{t0}-{t1}', 'epsilon': EPSILON,
                'n_primary': base_exc['n_included'],
                'n_with_epsilon': eps_exc['n_included'],
                'n_recovered': eps_exc['n_included'] - base_exc['n_included'],
                'max_abs_change_on_shared': float((b - e).abs().max()),
                'note': 'countries recovered by epsilon have a level of exactly 0, so their log '
                        'change is an artefact of the constant, not an estimate',
            })
    return pd.DataFrame(rows)


# ====================================================================================================
# summaries
# ====================================================================================================
def attach_weights(dec, pop):
    return dec.merge(pop, on='gid_0', how='left')


def category_summary(dec, groupby=None):
    """counts and shares per category, unweighted and weighted by adult female population"""
    keys = ['indicator', 'interval'] + (groupby or [])
    labels = {c['code']: c['label'] for c in CODEBOOK}

    frames = []
    for key_vals, block in dec.groupby(keys):
        total_n = len(block)
        total_w = block['pop_female'].sum()
        for code in sorted(labels):
            sub = block[block['category'] == code]
            row = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
            row.update({
                'category': code, 'category_label': labels[code],
                'n_countries': len(sub),
                'share_unweighted': len(sub) / total_n if total_n else np.nan,
                'share_pop_weighted': (sub['pop_female'].sum() / total_w) if total_w else np.nan,
                'parity_improved_without_access_gain': code in PARITY_WITHOUT_ACCESS,
            })
            frames.append(row)
    return pd.DataFrame(frames)


def extremes(dec, n=20):
    """largest parity improvements and deteriorations over the primary interval"""
    block = dec[dec['interval'] == f'{PRIMARY[0]}-{PRIMARY[1]}']
    cols = ['indicator', 'gid_0', 'country', 'region', 'delta_log_ggi',
            'female_component', 'male_component', 'category_label',
            'parity_direction', 'female_access_direction',
            'parity_improved_without_access_gain']
    rows = []
    for indicator in INDICATORS:
        b = block[block['indicator'] == indicator].sort_values('delta_log_ggi', ascending=False)
        rows.append(b.head(n)[cols].assign(rank_type='largest parity improvement'))
        rows.append(b.tail(n)[cols].assign(rank_type='largest parity deterioration')
                    .sort_values('delta_log_ggi'))
    return pd.concat(rows, ignore_index=True)


def regional_aggregate(levels, pop):
    """
    decomposition of the regional aggregate, on population-weighted regional levels

    Regional female level is the female-population-weighted mean of country female levels, and
    likewise for men — so each sex is aggregated on its own denominator. Weights are fixed at the
    base year, which keeps the identity exact at regional level too.
    """
    merged = levels.merge(pop, on='gid_0', how='inner')
    rows = []
    for indicator in INDICATORS:
        f_col, m_col = f'{indicator}_women', f'{indicator}_men'
        if params.level_ceiling_ctrl:
            # cap per country before weighting, matching the country-level treatment
            merged = merged.copy()
            merged[f_col] = np.minimum(merged[f_col], merged[m_col])
        for t0, t1 in [PRIMARY] + EXTRA_INTERVALS:
            for region, block in merged.groupby('region'):
                b = block[block['year'].isin([t0, t1])]
                usable = b.groupby('gid_0').filter(
                    lambda g: len(g) == 2 and g[[f_col, m_col]].gt(0).all().all())
                if usable.empty:
                    continue
                agg = {}
                for yr in (t0, t1):
                    y = usable[usable['year'] == yr]
                    agg[('f', yr)] = np.average(y[f_col], weights=y['pop_female'])
                    agg[('m', yr)] = np.average(y[m_col], weights=y['pop_male'])
                fc = np.log(agg[('f', t1)]) - np.log(agg[('f', t0)])
                mc = -(np.log(agg[('m', t1)]) - np.log(agg[('m', t0)]))
                rows.append({
                    'indicator': indicator, 'interval': f'{t0}-{t1}', 'region': region,
                    'n_countries': usable['gid_0'].nunique(),
                    'female_level_t0': agg[('f', t0)], 'female_level_t1': agg[('f', t1)],
                    'male_level_t0': agg[('m', t0)], 'male_level_t1': agg[('m', t1)],
                    'ggi_t0': agg[('f', t0)] / agg[('m', t0)],
                    'ggi_t1': agg[('f', t1)] / agg[('m', t1)],
                    'female_component': fc, 'male_component': mc,
                    'delta_log_ggi': fc + mc,
                    'parity_direction': 'improving' if fc + mc > TAU else
                                        'worsening' if fc + mc < -TAU else 'stable',
                    'female_access_direction': 'expanding' if fc > TAU else
                                               'contracting' if fc < -TAU else 'stable',
                })
    return pd.DataFrame(rows)


# ====================================================================================================
def verify(dec):
    """the identity must hold to machine precision, or the decomposition is not exact"""
    worst = dec['identity_residual'].abs().max()
    if not (worst <= TOLERANCE):
        raise AssertionError(f'decomposition identity violated: max |residual| = {worst:.3e}')
    return pd.DataFrame([{
        'check': 'delta_log_ggi == female_component + male_component',
        'tolerance': TOLERANCE,
        'max_abs_residual': worst,
        'n_rows': len(dec),
        'passed': True,
    }])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = f'{datetime.now():%Y%m%d}'

    levels = load_levels()
    pop = load_population()

    dec, exclusions = build_all(levels)
    identity = verify(dec)
    dec = attach_weights(dec, pop)

    ceiling_report = (dec.groupby(['indicator', 'interval'], as_index=False)
                      .agg(n=('ceiling_binding', 'size'),
                           n_ceiling_binding=('ceiling_binding', 'sum'))
                      .assign(pct_ceiling_binding=lambda d: d['n_ceiling_binding'] / d['n'] * 100,
                              level_ceiling_ctrl=params.level_ceiling_ctrl))

    outputs = {
        'codebook': pd.DataFrame(CODEBOOK).sort_values('code', ignore_index=True),
        'ceiling_report': ceiling_report,
        'country_period': dec,
        'exclusions': exclusions,
        'sensitivity': sensitivity(levels),
        'identity_check': identity,
        'category_global': category_summary(dec),
        'category_regional': category_summary(dec, ['region']),
        'extremes': extremes(dec),
        'regional_aggregate': regional_aggregate(levels, pop),
    }
    for name, frame in outputs.items():
        path = OUTDIR / f'{PREFIX}_{name}_{stamp}.csv'
        frame.to_csv(path, index=False)
        print(f'  {path.relative_to(params.ROOT)}  ({len(frame)} rows)')

    print(f'\nlevel_ceiling_ctrl = {params.level_ceiling_ctrl}  '
          f'(female level capped at the male level; parity is the ceiling)')
    print(f'identity: max |residual| = {identity["max_abs_residual"].iloc[0]:.2e} '
          f'over {len(dec)} country-periods — exact\n')
    print(ceiling_report[ceiling_report['interval'].isin(
        [f'{PRIMARY[0]}-{PRIMARY[1]}'] + [f'{a}-{b}' for a, b in EXTRA_INTERVALS])].round(2).to_string(index=False))
    print()
    print(exclusions[exclusions['interval'].isin(
        [f'{PRIMARY[0]}-{PRIMARY[1]}'] + [f'{a}-{b}' for a, b in EXTRA_INTERVALS])].to_string(index=False))
    print()
    primary = outputs['category_global']
    print(primary[primary['interval'] == f'{PRIMARY[0]}-{PRIMARY[1]}'].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
