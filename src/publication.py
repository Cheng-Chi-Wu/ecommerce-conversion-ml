"""Remove machine-specific paths before writing public reports."""
import re
from pathlib import Path


def sanitize_text(value, project_root=None):
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    # Specific locations must be replaced before their parent directory.
    replacements = [(root / '.venv', '<LOCAL_ENV>'),
                    (root / 'data/raw', '<LOCAL_DATA_DIR>/raw'),
                    (root / 'data/processed', '<LOCAL_DATA_DIR>/processed'),
                    (root, '<PROJECT_ROOT>')]
    for location, placeholder in replacements:
        value = value.replace(str(location), placeholder)
    token = r'[^\s\"\'<>(),\[\]\\]+'
    value = re.sub(r'/(?:Users|home)/' + token + r'/\.venv', '<LOCAL_ENV>', value)
    value = re.sub(r'/(?:Users|home)/' + token, '<LOCAL_DATA_DIR>', value)
    value = re.sub(r'~/' + token, '<LOCAL_DATA_DIR>', value)
    value = re.sub(r'(?:/opt/homebrew|/usr/local|/System/Volumes/Preboot/Cryptexes)' +
                   token, lambda match: '<LOCAL_ENV>/' + match.group().rsplit('/', 1)[-1], value)
    value = re.sub(r'/(?:private/)?(?:tmp|var/folders)/' + token, '<LOCAL_ENV>', value)
    value = re.sub(r'[A-Za-z]:\\Users\\[^\s\"\'<>(),\[\]]+', '<LOCAL_DATA_DIR>', value)
    return value


def sanitize_public_data(value):
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {sanitize_text(key): sanitize_public_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_public_data(item) for item in value]
    return value
