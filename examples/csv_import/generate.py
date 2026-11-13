"""Deterministic, fictional one-year CSV fixtures. Run from the repository root."""
from __future__ import annotations

import csv
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fund_analytics import db

CENT = Decimal('0.01')


def money(value):
    return format(Decimal(value).quantize(CENT), '.2f')


def records():
    """All amounts are generated with decimal arithmetic, without randomness."""
    clients, flows, distributions, fund = [], [], [], []
    opening = {f'T{i:03}': Decimal(100000 + i * 1000) for i in range(1, 13)}
    balances = opening.copy()
    price = Decimal(1)
    day = date(2025, 1, 1)
    while day.year == 2025:
        before = sum(balances.values())
        today_flows, today_distributions = [], []
        for i, client in enumerate(balances, 1):
            amount = None
            if day.day == 5 + i % 6:
                amount = Decimal(1000 + i * 25 + day.month * 10)
            elif day.day == 20 + i % 6:
                amount = -Decimal(500 + i * 15 + day.month * 5)
            if amount is not None:
                balances[client] += amount
                row = dict(transaction_id=len(flows) + 1, trade_date=day.isoformat(),
                           cash_flow_date=day.isoformat(), cash_flow_status='Settled',
                           client_id=client, flow_gbp=money(amount),
                           client_aum_after_trade_gbp=money(balances[client]))
                flows.append(row)
                today_flows.append(amount)
        if day.month in (3, 6, 9, 12) and day.day == 28:
            for client in balances:
                payment = (balances[client] * Decimal('.001')).quantize(CENT)
                balances[client] -= payment
                distributions.append(dict(distribution_id=len(distributions) + 1,
                    ex_distribution_date=day.isoformat(), payment_date=day.isoformat(),
                    client_id=client, distribution_gbp=money(payment), payment_status='Paid'))
                today_distributions.append(payment)
            price *= Decimal('.999')
        subs = sum((x for x in today_flows if x > 0), Decimal(0))
        reds = sum((-x for x in today_flows if x < 0), Decimal(0))
        paid = sum(today_distributions, Decimal(0))
        row = {column: money(0) for column in db.EXPECTED['fund_daily'].split(',')}
        row.update(date=day.isoformat(), opening_aum_gbp=money(before),
                   closing_aum_gbp=money(sum(balances.values())), active_client_count=12,
                   transaction_count=len(today_flows), income_unit_price_gbp=f'{price:.10f}',
                   accumulation_unit_price_gbp='1.0000000000',
                   income_distributions_declared_gbp=money(paid),
                   income_distributions_paid_gbp=money(paid),
                   net_investor_cash_flow_gbp=money(subs - reds - paid))
        for basis in ('trade', 'cash'):
            row[f'subscriptions_{basis}_date_gbp'] = money(subs)
            row[f'redemptions_{basis}_date_gbp'] = money(reds)
        row['net_flow_trade_date_gbp'] = money(subs - reds)
        row['net_client_flow_cash_date_gbp'] = money(subs - reds)
        fund.append(row)
        day += timedelta(days=1)
    sectors = ('Education', 'Health', 'Community')
    types = ('Charitable trust', 'Foundation', 'Association')
    for i, client in enumerate(balances, 1):
        clients.append(dict(client_id=client, client_name=f'Example Charity {i:02}',
            client_sector=sectors[(i - 1) % 3], client_type=types[(i - 1) % 3],
            country='England', region='Example region', unit_type='Income',
            account_open_date='2024-01-01', account_close_date='', valuation_date='2025-12-31',
            client_aum_gbp=money(balances[client]), opening_valuation_date='2024-12-31',
            opening_client_aum_gbp=money(opening[client])))
    return dict(clients=clients, client_flows=flows, income_distributions=distributions, fund_daily=fund)


def generate(destination=None):
    destination = Path(destination or Path(__file__).resolve().parent)
    tables = records()
    for variant in ('complete_2025', 'missing_sector_2025'):
        directory = destination / variant
        directory.mkdir(parents=True, exist_ok=True)
        for table, rows in tables.items():
            columns = db.EXPECTED[table].split(',')
            if variant == 'missing_sector_2025' and table == 'clients':
                columns.remove('client_sector')
            with (directory / db.FILES[table]).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    return destination


if __name__ == '__main__':
    print(generate())
