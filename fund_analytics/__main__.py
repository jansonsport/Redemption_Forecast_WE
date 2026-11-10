"""Local command line entry point."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Fund flow analytics pipeline')
    parser.add_argument('command', choices=['ingest', 'forecast', 'horizon-backtest', 'monthly-forecast', 'daily-forest-backtest', 'benchmark'])
    args = parser.parse_args()
    if args.command == 'ingest':
        from .db import ingest
        print(json.dumps(ingest(), default=str, indent=2))
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
    elif args.command == 'daily-forest-backtest':
        from .forecast import run_daily_forest_backtest
        summary, _, models = run_daily_forest_backtest()
        print('Selected random forest settings:')
        print(models.to_string(index=False))
        print(summary.to_string(index=False))
    else:
        from .benchmark import run
        run()


if __name__ == '__main__':
    main()
