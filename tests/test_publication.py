"""Public artifacts must not contain local paths; detector strings are rules."""
import json
import subprocess
from pathlib import Path

from src.common import ROOT, save_json
from src.publication import sanitize_text

# Construct the literal markers so a raw source scan cannot mistake rules for leaks.
LOCAL_MARKERS = ('/' + 'Users' + '/', 'sm' + 'arkwu', '~/' + 'Downloads')


def test_sanitization_preserves_errors_and_numeric_results(tmp_path):
    root = Path(LOCAL_MARKERS[0]) / 'tester' / 'project'
    text = (f'dlopen({root}/.venv/lib/xgboost.dylib): missing libomp; '
            f'data={root}/data/raw/events.csv; project={root}/src/modeling.py; '
            'search=/opt/homebrew/opt/libomp/lib/libomp.dylib')
    sanitized = sanitize_text(text, project_root=root)
    assert '<LOCAL_ENV>/lib/xgboost.dylib' in sanitized
    assert '<LOCAL_DATA_DIR>/raw/events.csv' in sanitized
    assert '<PROJECT_ROOT>/src/modeling.py' in sanitized
    assert 'missing libomp' in sanitized
    assert sanitize_text(sanitized, project_root=root) == sanitized
    assert not any(marker in sanitized for marker in LOCAL_MARKERS)
    target = tmp_path / 'report.json'
    save_json(target, {'test_ap': 0.2392221350609786, 'positives': 430,
                       'errors': [text], 'download': LOCAL_MARKERS[2] + '/archive/'})
    actual = json.loads(target.read_text())
    assert actual['test_ap'] == 0.2392221350609786
    assert actual['positives'] == 430
    assert actual['download'] == '<LOCAL_DATA_DIR>'
    assert not any(marker in target.read_text() for marker in LOCAL_MARKERS)


def test_public_files_have_no_local_paths():
    # Include new, non-ignored files as well as tracked files before staging.
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    leaks = []
    for name in set(names.decode().split('\0')) - {''}:
        path = ROOT / name
        if not path.is_file():
            continue  # A staged deletion has no current public artifact.
        text = path.read_bytes().decode('utf-8', errors='ignore')
        if any(marker in text for marker in LOCAL_MARKERS) or str(ROOT) in text:
            leaks.append(name)
    assert not leaks, f'Local paths in public files: {sorted(leaks)}'
