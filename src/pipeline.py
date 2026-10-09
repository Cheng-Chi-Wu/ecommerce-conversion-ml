"""CLI: python -m src.pipeline [all|ingest|sql|features|train|report]."""
import argparse

from .common import PROCESSED, REPORTS, FIGURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='all',
                        choices=['all', 'ingest', 'sql', 'features', 'train', 'report'])
    args = parser.parse_args()
    for folder in [PROCESSED, REPORTS, FIGURES]:
        folder.mkdir(parents=True, exist_ok=True)
    if args.stage in ['all', 'ingest']:
        from .ingest import build_database
        build_database()
    if args.stage in ['all', 'sql']:
        from .analytics import run_analytics
        run_analytics()
    if args.stage in ['all', 'features']:
        from .features import build_features
        build_features()
    if args.stage in ['all', 'train']:
        from .modeling import train_models
        train_models()
    if args.stage in ['all', 'report']:
        from .reporting import write_results
        write_results()


if __name__ == '__main__':
    main()
