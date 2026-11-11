"""Account-exposure helper shared with the client-relationship study (from handoff analysis_helpers.py)."""
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
