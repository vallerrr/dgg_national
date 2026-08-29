"""
# Created by valler at 18/08/2026
Feature: rebuild `offline_predictors.csv` from source, and report honestly how much of the shipped
file it reproduces.

The shipped file was assembled by hand in May 2024 and no build script survived. This reconstructs
what can be reconstructed and measures the rest, so the provenance is code rather than memory.

WHAT A REBUILD CAN AND CANNOT REPRODUCE — measured against the shipped file, not assumed:

  UNDP (30 vars)        EXACT. The shipped values come from HDR 2023-24, which is still
                        downloadable as a pinned file: 35,327 of 35,327 overlapping values match.
                        Pinning the vintage is what makes this reproducible; HDR 2025 matches only
                        17%, because UNDP revises the entire back-series every edition.

  World Bank (15 vars)  NOT REPRODUCIBLE. The API serves only the current vintage and the World
                        Bank revises history too — today's API matches ~7% of the shipped
                        gdp_pcap, median relative revision ~1%. There is no pinned-vintage
                        endpoint, so a rebuild necessarily produces different numbers for years
                        already in the file.

  UNESCO, WEF/QoG,      NOT AUTOMATED. These need a manual download (registration, bulk archives,
  V-Dem, BICC (13)      or PDF-parsed tables). The steps are in `MANUAL_SOURCES` below; the script
                        carries the shipped columns through untouched and says so.

So this is a *reproduction check*, not a drop-in replacement. If the aim is only to extend the
file forward, use `05_offline_predictors_update.py`, which never touches an existing value.

    python src/data_creation/06_offline_predictors_build.py --report-only
    python src/data_creation/06_offline_predictors_build.py
"""
import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import params  # noqa: E402

# reuse the fetchers and the variable maps rather than restating them
_spec = importlib.util.spec_from_file_location(
    'updater', Path(__file__).with_name('05_offline_predictors_update.py'))
updater = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(updater)

# The vintage that reproduces the shipped file. Do not "upgrade" this to be helpful: changing it
# rewrites every historical value in the panel.
HDR_PINNED = ('HDR23-24', 'https://hdr.undp.org/sites/default/files/2023-24_HDR/'
                          'HDR23-24_Composite_indices_complete_time_series.csv')

MANUAL_SOURCES = {
    'gggi_ggi, gggi_eas, gggi_hss, gggi_pes, gggi_pos': (
        'World Economic Forum Global Gender Gap Report, via the Quality of Government dataset '
        '(the WEF publishes PDF; QoG parses it). Download the QoG Standard time-series from '
        'https://www.gu.se/en/quality-government/qog-data and take the wef_* columns.'),
    'vdem_gender': (
        'V-Dem. Register at https://v-dem.net/data/the-v-dem-dataset/ and take the '
        'women political empowerment index from the Country-Year Full+Others file.'),
    'bicc_gmi': (
        'Bonn International Centre for Conflict Studies Global Militarisation Index, '
        'https://gmi.bicc.de/ — annual XLSX, no API.'),
    'une_girlglsf, une_girlglsm, une_girlglst, une_girlgpf, une_girlgpm, une_girlgpt': (
        'UNESCO Institute for Statistics bulk download, https://uis.unesco.org/bdds — '
        'the SDG or OPRI archive, gross intake/enrolment by sex.'),
    'pov_hr': (
        'World Bank SI.POV.DDAY. Excluded from the automated path: the poverty line was rebased '
        'to $3.00/day 2021 PPP, so the current series is not the same definition as the shipped '
        'column.'),
}


def build_undp(cache):
    """the UNDP block, from the pinned HDR vintage"""
    label, url = HDR_PINNED
    path = cache / f'{label.lower()}_composite.csv'
    if not path.exists():
        print(f'  downloading {label} (pinned) ...')
        updater.download(url, path)
    saved = updater.HDR_LATEST
    updater.HDR_LATEST = (label, url)          # reuse the melt logic on the pinned file
    try:
        return updater.fetch_hdr(cache)
    finally:
        updater.HDR_LATEST = saved


def compare(shipped, rebuilt, variables, label):
    """per-variable agreement between the rebuild and the shipped file"""
    key = ['iso3', 'year']
    joined = shipped[key + [v for v in variables if v in shipped.columns]].merge(
        rebuilt, on=key, how='inner', suffixes=('_shipped', '_rebuilt'))

    rows = []
    for var in variables:
        a, b = f'{var}_shipped', f'{var}_rebuilt'
        if a not in joined.columns or b not in joined.columns:
            continue
        both = joined[a].notna() & joined[b].notna()
        if not both.any():
            continue
        x, y = joined.loc[both, a], joined.loc[both, b]
        exact = np.isclose(x, y, rtol=0, atol=1e-9)
        denom = y.abs().replace(0, np.nan)
        rows.append({'variable': var, 'source': label, 'n_compared': int(both.sum()),
                     'pct_exact': 100 * float(exact.mean()),
                     'median_rel_diff': float(((x - y).abs() / denom).median()),
                     'max_abs_diff': float((x - y).abs().max())})
    return pd.DataFrame(rows)


def main(report_only):
    cache = params.PROCESSED / 'offline_predictors_cache'
    cache.mkdir(parents=True, exist_ok=True)

    shipped = pd.read_csv(params.RAW / 'offline_predictors.csv')
    print(f'shipped: {len(shipped):,} rows, {len(shipped.columns)} columns, '
          f'years {int(shipped.year.min())}-{int(shipped.year.max())}')

    print('\nUNDP (pinned vintage):')
    undp = build_undp(cache)
    print('\nWorld Bank (current vintage — no pinned endpoint exists):')
    worldbank = updater.fetch_worldbank(cache, datetime.now().year)

    report = pd.concat([
        compare(shipped, undp, updater.UNDP_VARS, HDR_PINNED[0]),
        compare(shipped, worldbank, list(updater.WB_VARS), 'WDI (current)'),
    ], ignore_index=True).sort_values(['source', 'pct_exact'], ascending=[True, False])

    print('\n' + '=' * 92)
    print('REPRODUCTION CHECK — does the rebuild recover the shipped values?')
    print('=' * 92)
    print(report.round(4).to_string(index=False))

    for source, block in report.groupby('source'):
        print(f'\n  {source}: {block["pct_exact"].mean():.1f}% of values reproduced exactly '
              f'(mean over {len(block)} variables)')

    print('\n' + '=' * 92)
    print('NOT AUTOMATED — these columns are carried through from the shipped file')
    print('=' * 92)
    for columns, how in MANUAL_SOURCES.items():
        print(f'\n  {columns}\n    {how}')

    if report_only:
        print('\n[report-only] nothing written.')
        return

    # the rebuild: pinned UNDP + current World Bank, manual columns carried through
    rebuilt = shipped[['iso3', 'year', 'country']].drop_duplicates(['iso3', 'year']).copy()
    for frame in (undp, worldbank):
        rebuilt = rebuilt.merge(frame, on=['iso3', 'year'], how='left')
    carried = [c for c in shipped.columns if c not in rebuilt.columns]
    rebuilt = rebuilt.merge(shipped[['iso3', 'year'] + carried], on=['iso3', 'year'], how='left')

    for ratio, (f_col, m_col) in updater.RATIOS.items():
        if f_col in rebuilt.columns and m_col in rebuilt.columns:
            rebuilt[ratio] = rebuilt[f_col] / rebuilt[m_col].replace(0, np.nan)

    rebuilt = rebuilt.sort_values(['iso3', 'year'], ignore_index=True)
    rebuilt = rebuilt[[c for c in shipped.columns if c in rebuilt.columns]]

    stamp = f'{datetime.now():%Y%m%d}'
    out = params.RAW / f'offline_predictors_rebuilt_{stamp}.csv'
    rebuilt.to_csv(out, index=False)
    report.to_csv(params.RAW / f'offline_predictors_rebuild_report_{stamp}.csv', index=False)
    print(f'\nwrote {out}  ({len(rebuilt):,} rows)')
    print('This is NOT the shipped file: its World Bank columns are a newer vintage and its '
          'history differs. Read the reproduction table above before using it for anything.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report-only', action='store_true',
                    help='print the reproduction check without writing a rebuilt file')
    main(ap.parse_args().report_only)
