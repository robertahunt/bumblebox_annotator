"""Small read-only lab probe, isolated from the GUI and model imports."""

import json
import os
import sys

from core.project_sync import check_destination


def check_lab_access(configuration):
    root = check_destination(configuration['root'], configuration['root_id'])
    # Reading the marker alone does not prove directory listing is permitted.
    with os.scandir(root) as entries:
        next(entries, None)
    return {'connected': True}


def main():
    try:
        result = check_lab_access(json.load(sys.stdin))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {'connected': False, 'error': str(exc)}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
