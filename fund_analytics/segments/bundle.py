"""Offline build of the Segments & stress page data (`python -m fund_analytics segments`).

Runs the ported study pipeline once, writes its `11_*.csv` tables and a single pickle bundle under
data/clean/segments, and copies the study notebook HTML for static serving. Recomputes nothing
statistical itself. The bundle carries the source fingerprint so the page can refuse stale data.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import db
from . import pipeline as S

VERSION = 'segments-v1'
OUT = S.OUT
BUNDLE = OUT / 'bundle.pkl'
NOTEBOOK_SRC = S.ROOT / 'handoff/segments_stress/analysis/11_segments_correlation_stress.html'
STATIC_DIR = S.ROOT / 'static/segments'


def source_fingerprint(raw_dir: Path = db.RAW_DIR) -> str:
    """Same formula as db.ingest, so the page can compare it with the latest import."""
    hashes = {}
    for filename in db.FILES.values():
        with (Path(raw_dir) / filename).open('rb') as stream:
            hashes[filename] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashlib.sha256(json.dumps([db.SCHEMA_VERSION, hashes], sort_keys=True).encode()).hexdigest()


def segment_order(profiles: pd.DataFrame) -> list[str]:
    """Largest segment by AUM first, Rare traders last: the fixed order behind every segment colour."""
    ranked = profiles.sort_values('aum_gbp', ascending=False).segment
    return [s for s in ranked if s != 'Rare traders'] + ['Rare traders']


def build(cfg=S.CONFIG) -> dict:
    started = time.perf_counter()
    out = S.run(cfg, write=True)
    if not out['checks'].passed.all():
        failed = out['checks'].loc[~out['checks'].passed, 'check'].tolist()
        raise ValueError(f'Segments pipeline data checks failed: {failed}')
    bundle = make_bundle(out, cfg)
    meta = bundle['meta']
    OUT.mkdir(parents=True, exist_ok=True)
    meta['duration_seconds'] = time.perf_counter() - started
    tmp = BUNDLE.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(BUNDLE)
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str))
    if NOTEBOOK_SRC.exists():
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(NOTEBOOK_SRC, STATIC_DIR / NOTEBOOK_SRC.name)
    return meta


def make_bundle(out: dict, cfg=S.CONFIG) -> dict:
    """The page's data contract: pipeline outputs plus client attributes and fund daily series."""
    out = {k: v for k, v in out.items() if k != 'rate'}  # the engine recomputes rates from seg_aum / seg_net
    c = cfg['clients']
    clients = pd.read_csv(S.RAW / cfg['files']['clients'], parse_dates=[c['open'], c['close']])
    clients = clients[[c['id'], c['name'], c['sector'], c['type'], c['country'], c['region'], c['share_class'],
                       c['open'], c['close'], c['balance']]].set_index(c['id'])
    fu = cfg['fund']
    fund = pd.read_csv(S.RAW / cfg['files']['fund'], parse_dates=[fu['date']]).set_index(fu['date'])
    daily = pd.DataFrame({'price': fund[fu['prices']['Accumulation']], 'fund_sub': fund[fu['sub']],
                          'fund_red': fund[fu['red']], 'fund_open': fund[fu['open_aum']]}).reindex(out['seg_aum'].index)
    k = int(out['selection'].loc[out['selection'].chosen, 'k'].iloc[0])
    meta = dict(version=VERSION, source_fingerprint=source_fingerprint(), built_at=pd.Timestamp.now().isoformat(timespec='seconds'),
                k=k, k_auto=int(out['selection'].loc[out['selection'].auto_choice, 'k'].iloc[0]),
                ari_median=float(np.median(out['ari'])), config_name=cfg['name'],
                episodes={k_: list(v) for k_, v in cfg['named_episodes'].items()}, eligibility=cfg['eligibility'],
                floors={k_: list(v) for k_, v in cfg['floors'].items()}, buffers=cfg['buffers'],
                order=segment_order(out['profiles']), duration_seconds=None)
    return dict(meta=meta, out=out, clients=clients, daily=daily)


def load(path: Path = BUNDLE) -> dict:
    """Load a bundle written by build() on this machine (generated artifact, not untrusted input)."""
    with Path(path).open('rb') as stream:
        return pickle.load(stream)
