"""
# Created by valler at 18/08/2026
Feature: extend `offline_predictors.csv` forward without disturbing a single existing value.

The shipped file stops at 2022 for most variables, which is what blocks the 2023-2025 surveys from
entering the fitting panel (D39). Newer releases exist:

    hdi, gdi + 27 other UNDP indicators   HDR 2025          -> 2023
    gdp_pcap and the World Bank series    WDI API           -> 2025

THE RULE THIS SCRIPT ENFORCES: an existing non-null cell is never written. Only cells that are
empty *and* beyond the variable's current last year are filled, so everything at or below 2022
stays byte-identical. That is asserted before anything is written, not assumed.

WHY IT CANNOT SIMPLY REBUILD EVERYTHING (measured, not guessed):

    UNDP     HDR 2023-24 reproduces the shipped file exactly — 35,327 of 35,327 values.
             HDR 2025 reproduces 17% of them: it revises the whole back-series every edition
             (median HDI revision 0.008, median GNI per capita revision $1,191).
    World    The WDI API serves only the current vintage. It reproduces 7% of the shipped
    Bank     gdp_pcap values; median relative revision ~1%.

So appending is the only operation that preserves history. The cost is a VINTAGE-MIXED file: rows
up to 2022 are HDR 2023-24, rows from 2023 are HDR 2025, and a 2023 HDI is therefore not on quite
the same basis as a 2022 one. `06_offline_predictors_build.py` is the alternative — one consistent
vintage throughout, at the price of changing history. Neither is free; see doc/decisions_shared.md.

    python src/data_creation/05_offline_predictors_update.py --dry-run
    python src/data_creation/05_offline_predictors_update.py
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import params  # noqa: E402

HDR_LATEST = ('HDR25', 'https://hdr.undp.org/sites/default/files/2025_HDR/'
                       'HDR25_Composite_indices_complete_time_series.csv')
WB_API = ('https://api.worldbank.org/v2/country/all/indicator/{code}'
          '?format=json&per_page=25000&date=1960:{to}&source={source}')

# World Bank database ids. The default endpoint serves WDI only; Worldwide Governance Indicators
# live in a separate database and return an EMPTY payload without this, which is silent unless
# you check the row count.
WDI, WGI = 2, 3

# our column name -> HDR column stem. 29 of the 30 match directly; gni_pc is the exception.
UNDP_VARS = ['abr', 'coef_ineq', 'eys', 'eys_f', 'eys_m', 'gdi', 'gii', 'gni_pc_f', 'gni_pc_m',
             'gni_pc', 'hdi', 'hdi_f', 'hdi_m', 'ihdi', 'ineq_edu', 'ineq_inc', 'ineq_le', 'le',
             'le_f', 'le_m', 'lfpr_f', 'lfpr_m', 'mmr', 'mys', 'mys_f', 'mys_m', 'pr_f', 'pr_m',
             'se_f', 'se_m']
UNDP_STEM = {v: ('gnipc' if v == 'gni_pc' else v) for v in UNDP_VARS}

# our column name -> World Bank indicator code
WB_VARS = {
    'gdp_pcap': 'NY.GDP.PCAP.CD',
    'wdi_gert': 'SE.TER.ENRR', 'wdi_gertf': 'SE.TER.ENRR.FE', 'wdi_gertm': 'SE.TER.ENRR.MA',
    'wdi_litrad': 'SE.ADT.LITR.ZS', 'wdi_litradf': 'SE.ADT.LITR.FE.ZS',
    'wdi_litradm': 'SE.ADT.LITR.MA.ZS',
    'wdi_acel': 'EG.ELC.ACCS.ZS', 'wdi_fertility': 'SP.DYN.TFRT.IN',
    'wdi_unemp': 'SL.UEM.TOTL.ZS', 'wdi_unempf': 'SL.UEM.TOTL.FE.ZS',
    'wdi_unempm': 'SL.UEM.TOTL.MA.ZS',
    'wdi_internet': 'IT.NET.USER.ZS', 'hcpi_a': 'FP.CPI.TOTL.ZG',
    # WGI codes carry a GOV_WGI_ prefix in the API and live in database 3; the bare 'PV.EST'
    # returns an empty payload rather than an error.
    'wbgi_pve': 'GOV_WGI_PV.EST',
}
WB_SOURCE = {'wbgi_pve': WGI}       # everything else is WDI

# DELIBERATELY NOT UPDATED: `pov_hr`. The World Bank rebased the poverty line to $3.00/day in 2021
# PPP; the shipped column is on the older $2.15 (2017 PPP) definition. Appending the new series
# would splice two different definitions into one column and the break would be invisible
# downstream. Refresh it only alongside a decision about restating the history.

# derived columns: female/male ratios, recomputed only for rows this script adds
RATIOS = {'hdi_r': ('hdi_f', 'hdi_m'), 'le_r': ('le_f', 'le_m'), 'eys_r': ('eys_f', 'eys_m'),
          'mys_r': ('mys_f', 'mys_m'), 'gni_pc_r': ('gni_pc_f', 'gni_pc_m'),
          'se_r': ('se_f', 'se_m'), 'lfpr_r': ('lfpr_f', 'lfpr_m'), 'pr_r': ('pr_f', 'pr_m'),
          'une_girlglsr': ('une_girlglsf', 'une_girlglsm'),
          'une_girlgpr': ('une_girlgpf', 'une_girlgpm'),
          'wdi_gertr': ('wdi_gertf', 'wdi_gertm'),
          'wdi_litradr': ('wdi_litradf', 'wdi_litradm'),
          'wdi_unempr': ('wdi_unempf', 'wdi_unempm')}


def download(url, path, attempts=4):
    """
    fetch a URL to disk

    hdr.undp.org returns 403 to a bare urllib request, so a browser User-Agent is set. Cached on
    disk: these are the inputs to a reproducible build, not something to re-pull on every run.
    """
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                path.write_bytes(response.read())
            return
        except urllib.error.HTTPError as error:
            # the World Bank API returns a sporadic 400 under rapid sequential requests
            if attempt == attempts:
                raise
            print(f'    retry {attempt}/{attempts - 1} after HTTP {error.code}')
            time.sleep(2 * attempt)


def fetch_hdr(cache):
    """the HDR composite-indices time series, long-form"""
    label, url = HDR_LATEST
    path = cache / f'{label.lower()}_composite.csv'
    if not path.exists():
        print(f'  downloading {label} ...')
        download(url, path)
    # the file is latin-1, not UTF-8 — a default read_csv fails on it
    wide = pd.read_csv(path, low_memory=False, encoding='latin1')

    frames = []
    for var, stem in UNDP_STEM.items():
        cols = [c for c in wide.columns if re.fullmatch(rf'{re.escape(stem)}_(\d{{4}})', c)]
        if not cols:
            print(f'    ! {var}: no {stem}_<year> columns in {label}')
            continue
        long = wide.melt(id_vars='iso3', value_vars=cols, var_name='k', value_name=var)
        long['year'] = long['k'].str.extract(r'(\d{4})').astype(int)
        frames.append(long[['iso3', 'year', var]])

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=['iso3', 'year'], how='outer')
    return out


def fetch_worldbank(cache, to_year):
    """one World Bank indicator per column, via the API"""
    frames = []
    for var, code in WB_VARS.items():
        source = WB_SOURCE.get(var, WDI)
        path = cache / f'wb_{code}_src{source}.json'
        if not path.exists():
            print(f'  downloading {code} ({var}) ...')
            download(WB_API.format(code=code, to=to_year, source=source), path)
        payload = json.load(open(path))
        if len(payload) < 2 or not payload[1]:
            print(f'    ! {var}: empty response for {code}')
            continue
        frames.append(pd.DataFrame(
            [{'iso3': r['countryiso3code'], 'year': int(r['date']), var: r['value']}
             for r in payload[1] if r['countryiso3code']]))

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=['iso3', 'year'], how='outer')
    return out


def last_year_per_variable(current, variables):
    """the newest year each variable currently has a value for"""
    return {v: int(current.loc[current[v].notna(), 'year'].max())
            for v in variables if v in current.columns and current[v].notna().any()}


def extend(current, incoming, variables, source_label):
    """
    fill forward-only, empty-only cells

    Returns the merged frame and a per-variable report. A cell is written only when all three
    hold: the variable's year is beyond what the file already has, the existing cell is empty,
    and the incoming value is not null.
    """
    cutoffs = last_year_per_variable(current, variables)
    merged = current.merge(incoming, on=['iso3', 'year'], how='outer', suffixes=('', '_new'))

    report = []
    for var in variables:
        new_col = f'{var}_new'
        if new_col not in merged.columns:
            continue
        cutoff = cutoffs.get(var)
        if cutoff is None:
            merged.drop(columns=new_col, inplace=True)
            continue
        writable = (merged['year'] > cutoff) & merged[var].isna() & merged[new_col].notna()
        merged.loc[writable, var] = merged.loc[writable, new_col]
        report.append({'variable': var, 'source': source_label, 'was_through': cutoff,
                       'cells_added': int(writable.sum()),
                       'now_through': int(merged.loc[merged[var].notna(), 'year'].max())})
        merged.drop(columns=new_col, inplace=True)
    return merged, pd.DataFrame(report)


def recompute_ratios(frame, cutoff_by_ratio):
    """female/male ratios, for added rows only — existing ratio cells are left alone"""
    added = []
    for ratio, (f_col, m_col) in RATIOS.items():
        if ratio not in frame.columns or f_col not in frame.columns or m_col not in frame.columns:
            continue
        cutoff = cutoff_by_ratio.get(ratio)
        if cutoff is None:
            continue
        writable = (frame['year'] > cutoff) & frame[ratio].isna() \
            & frame[f_col].notna() & frame[m_col].notna() & (frame[m_col] != 0)
        frame.loc[writable, ratio] = frame.loc[writable, f_col] / frame.loc[writable, m_col]
        added.append({'variable': ratio, 'source': 'derived', 'was_through': cutoff,
                      'cells_added': int(writable.sum()),
                      'now_through': int(frame.loc[frame[ratio].notna(), 'year'].max())})
    return frame, pd.DataFrame(added)


def assert_history_intact(before, after, tol=1e-12):
    """
    every value the shipped file had must survive unchanged

    This is the promise of the whole script, so it is checked rather than trusted. Compared on the
    (iso3, year) key over every column they share.

    `tol` is a RELATIVE tolerance, and it exists for one reason: writing a float to CSV and reading
    it back is not always bit-exact. The shipped file stores some values at 16 significant figures
    where the double needs 17 (e.g. hcpi_a BGR 1973 = -1.100000014578155e-14), so a round trip can
    move the last bit. 19 values in this file are affected, all at magnitudes around 1e-14 — i.e.
    numerically zero. Anything above `tol` is a real change and aborts the run; the exact-match
    count is reported separately so the distinction stays visible rather than hidden by the
    tolerance.
    """
    key = ['iso3', 'year']
    shared = [c for c in before.columns if c in after.columns and c not in key]
    merged = before.merge(after, on=key, how='left', suffixes=('_before', '_after'))

    problems, n_exact, n_ulp, n_total = [], 0, 0, 0
    for col in shared:
        a, b = merged[f'{col}_before'], merged[f'{col}_after']
        had = a.notna()
        lost = int((had & b.isna()).sum())
        both = had & b.notna()
        n_total += int(both.sum())
        if pd.api.types.is_numeric_dtype(a):
            exact = both & np.isclose(a.where(both), b.where(both), rtol=0, atol=0, equal_nan=True)
            close = both & np.isclose(a.where(both), b.where(both), rtol=tol, equal_nan=True)
            changed = int((both & ~close).sum())
        else:
            exact = both & (a == b)
            changed = int((both & (a != b)).sum())
        n_exact += int(exact.sum())
        n_ulp += int((both & ~exact).sum()) - changed
        if lost or changed:
            problems.append({'column': col, 'values_lost': lost, 'values_changed': changed})
    return pd.DataFrame(problems), {'compared': n_total, 'exact': n_exact, 'round_trip': n_ulp}


def main(dry_run):
    cache = params.PROCESSED / 'offline_predictors_cache'
    cache.mkdir(parents=True, exist_ok=True)

    source = params.RAW / 'offline_predictors.csv'
    current = pd.read_csv(source)
    print(f'current file: {len(current):,} rows, {len(current.columns)} columns, '
          f'years {int(current.year.min())}-{int(current.year.max())}')
    before = current.copy()

    ratio_cutoffs = last_year_per_variable(current, list(RATIOS))

    print('\nUNDP:')
    updated, undp_report = extend(current, fetch_hdr(cache), UNDP_VARS, HDR_LATEST[0])
    print('\nWorld Bank:')
    updated, wb_report = extend(updated, fetch_worldbank(cache, datetime.now().year),
                                list(WB_VARS), 'WDI')
    updated, ratio_report = recompute_ratios(updated, ratio_cutoffs)

    updated = updated.sort_values(['iso3', 'year'], ignore_index=True)[list(before.columns)]

    report = pd.concat([undp_report, wb_report, ratio_report], ignore_index=True)
    report = report[report['cells_added'] > 0].sort_values('cells_added', ascending=False)

    print('\n' + '=' * 78)
    print('WHAT WOULD BE ADDED')
    print('=' * 78)
    print(report.to_string(index=False) if len(report) else '  nothing — already current')

    print('\n' + '=' * 78)
    print('HISTORY CHECK — every shipped value must survive unchanged')
    print('=' * 78)
    problems, counts = assert_history_intact(before, updated)
    if len(problems):
        print(problems.to_string(index=False))
        raise SystemExit('ABORTED: the update would alter existing values.')
    print(f'  {len(before):,} shipped rows intact; {counts["exact"]:,} of {counts["compared"]:,} '
          f'values bit-identical, {counts["round_trip"]} differing only in the last bit.')

    if dry_run:
        print('\n[dry-run] nothing written.')
        return

    stamp = f'{datetime.now():%Y%m%d}'
    out = params.RAW / f'offline_predictors_{stamp}.csv'
    updated.to_csv(out, index=False)

    # verify the FILE, not the frame: the promise is about what lands on disk
    problems, counts = assert_history_intact(before, pd.read_csv(out))
    if len(problems):
        print(problems.to_string(index=False))
        raise SystemExit(f'ABORTED: {out} does not preserve the shipped values.')
    print(f'  written file re-checked: {counts["exact"]:,} of {counts["compared"]:,} values '
          f'bit-identical, {counts["round_trip"]} differing only in the last bit (CSV round trip).')
    report.to_csv(params.RAW / f'offline_predictors_update_report_{stamp}.csv', index=False)
    print(f'\nwrote {out}  ({len(updated):,} rows)')
    print('Date-stamped, per doc/00_project_conventions.md §4 — the shipped file is untouched. Point '
          'params at the new one deliberately, not by default.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would be added and verify history, without writing')
    main(ap.parse_args().dry_run)
