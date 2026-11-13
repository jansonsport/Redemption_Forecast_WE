"""Small shared helpers for notebook-based client cash-flow analysis.

All periods use inclusive account boundaries and the same exposure policy as
Notebook 04. Original files and transaction dates are never changed.
"""
import numpy as np
import pandas as pd


def complete_period_mask(metadata, frequency, start, end):
    """True only for complete periods inside account life AND source coverage."""
    periods = pd.period_range(start, end, freq=frequency)
    starts = periods.start_time.to_numpy()
    ends = periods.end_time.normalize().to_numpy()
    opens = metadata.account_open_date.to_numpy()
    closes = metadata.account_close_date.fillna(end).to_numpy()
    eligible = (starts[:, None] >= opens[None, :]) & (ends[:, None] <= closes[None, :])
    eligible &= (starts[:, None] >= pd.Timestamp(start).to_datetime64())
    eligible &= (ends[:, None] <= pd.Timestamp(end).to_datetime64())
    return pd.DataFrame(eligible, index=periods, columns=metadata.client_id)


def panel_from_aggregated(aggregated, metadata, frequency, start, end):
    """Align an event aggregate once; zero only fully exposed, event-absent cells."""
    mask = complete_period_mask(metadata, frequency, start, end)
    panel = aggregated.reindex(index=mask.index, columns=mask.columns).fillna(0).where(mask)
    return panel, mask


def pair_correlations(panel, minimum):
    """Pearson coefficients, overlap counts and each unordered pair once.

    Vectorised extraction avoids repeatedly slicing a large panel for each pair.
    The caller checks that all observed values are finite. With that condition,
    sufficient-overlap NaNs from pandas corr indicate zero variance in overlap.
    """
    assert np.isfinite(panel.to_numpy()[panel.notna().to_numpy()]).all()
    correlation = panel.corr(method='pearson', min_periods=minimum)
    observed = panel.notna().astype('int64')
    overlap = observed.T.dot(observed)
    left, right = np.triu_indices(panel.shape[1], k=1)
    values = correlation.to_numpy()[left, right]
    counts = overlap.to_numpy()[left, right]
    identifiers = panel.columns.to_numpy()
    pairs = pd.DataFrame({'client_a': identifiers[left], 'client_b': identifiers[right],
                          'overlapping_periods': counts, 'correlation': values})
    pairs['status'] = np.select([counts < minimum, np.isfinite(values)],
                               ['insufficient overlap', 'defined'], default='zero variance in overlap')
    assert pairs.status.eq('defined').equals(pairs.correlation.notna())
    return correlation, overlap, pairs


def sample_sectors(eligible, per_sector, seed):
    """Use a sorted population and identical fixed sample size per sector.

    Seeding each sector independently reproduces Notebook 04's Faith sample for
    seed 42. No activity or correlation values enter sampling.
    """
    samples = []
    for sector, population in eligible.groupby('client_sector', sort=True):
        assert len(population) >= per_sector
        samples.append(population.sort_values('client_id').sample(per_sector, random_state=seed))
    return pd.concat(samples).sort_values(['client_sector', 'client_id']).reset_index(drop=True)
