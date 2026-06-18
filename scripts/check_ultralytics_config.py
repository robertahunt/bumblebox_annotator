#!/usr/bin/env python3
"""Check Ultralytics configuration for validation plot settings"""

from ultralytics import __version__
from ultralytics.utils import DEFAULT_CFG

print(f'Ultralytics version: {__version__}')
print('\nValidation and plot-related defaults:')
for k in sorted(vars(DEFAULT_CFG).keys()):
    v = getattr(DEFAULT_CFG, k)
    if 'val' in k.lower() or 'plot' in k.lower():
        print(f'  {k}: {v}')
