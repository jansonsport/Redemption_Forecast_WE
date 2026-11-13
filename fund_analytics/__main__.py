"""Local command line entry point."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Fund flow analytics pipeline')
    parser.add_argument('command', choices=['ingest', 'forecast', 'horizon-backtest', 'monthly-forecast', 'daily-forest-backtest', 'history-window-backtest', 'history-window-intervals', 'performance-backtest', 'selected-forecast', 'segments', 'benchmark'])
    parser.add_argument('--raw-dir', type=str, help='Folder of canonical CSV inputs (ingest only)')
    parser.add_argument('--scan-rows', type=int, default=100, help='Maximum records scanned for headers')
    parser.add_argument('--source', action='append', default=[], metavar='TABLE=PATH')
    parser.add_argument('--header-row', action='append', default=[], metavar='TABLE=N')
    args = parser.parse_args()
    def pairs(values, convert=str):
        result = {}
        for value in values:
            if '=' not in value:
                parser.error('Overrides must be TABLE=VALUE')
            table, entry = value.split('=', 1)
            if table in result:
                parser.error(f'Duplicate override for {table}')
            try:
                result[table] = convert(entry)
            except ValueError:
                parser.error(f'Invalid override {value}')
        return result
    if args.command not in ('ingest', 'benchmark'):
        from .analytics import require_feature
        require_feature('segments' if args.command == 'segments' else 'forecast')
    if args.command == 'ingest':
        from .db import ingest
        options = dict(scan_rows=args.scan_rows, sources=pairs(args.source), header_rows=pairs(args.header_row, int))
        if args.raw_dir:
            options['raw_dir'] = args.raw_dir
        print(json.dumps(ingest(**options), default=str, indent=2))
    elif args.command == 'forecast':
        from .forecast import run_forecasts
        run_forecasts()
    elif args.command == 'horizon-backtest':
        from .forecast import run_multi_horizon_backtest
        _, summary = run_multi_horizon_backtest()
        print(summary.to_string(index=False))
    elif args.command == 'monthly-forecast':
        from .monthly_forecast import run
        result = run()
        print(json.dumps(result['metadata'], indent=2))
    elif args.command == 'selected-forecast':
        from .selected_forecasts import run
        print(run()['models'].to_string(index=False))
    elif args.command == 'performance-backtest':
        from .performance_backtest import run
        print(run()['comparisons'].query("period == 'overall'").to_string(index=False))
    elif args.command == 'history-window-intervals':
        from .history_intervals import run
        result = run()
        print(result['metrics'].to_string(index=False))
    elif args.command == 'history-window-backtest':
        from .history_backtest import run
        print(run()['metrics'].to_string(index=False))
    elif args.command == 'daily-forest-backtest':
        from .forecast import run_daily_forest_backtest
        summary, _, models = run_daily_forest_backtest()
        print('Selected random forest settings:')
        print(models.to_string(index=False))
        print(summary.to_string(index=False))
    elif args.command == 'segments':
        from .segments.bundle import build
        print(json.dumps(build(), indent=2, default=str))
    else:
        from .benchmark import run
        run()


if __name__ == '__main__':
    main()
