"""Stable annotation category keys and appearance-based brood labels."""

BASE_CATEGORIES = ('bee', 'hive', 'chamber', 'pollen')
STANDARD_BROOD_CATEGORIES = ('brood_early', 'brood_early_middle', 'brood_middle',
                             'brood_middle_late', 'brood_late')
QUEEN_BROOD_CATEGORIES = ('queen_brood_middle', 'queen_brood_middle_late', 'queen_brood_late')
BROOD_CATEGORIES = STANDARD_BROOD_CATEGORIES + QUEEN_BROOD_CATEGORIES
# Append new categories so existing project/COCO category IDs remain stable.
CATEGORIES = BASE_CATEGORIES + BROOD_CATEGORIES + ('nectar',)
BROOD_MODEL_LABEL = f'Brood ({len(BROOD_CATEGORIES)} appearance classes)'
CATEGORY_MENU_GROUPS = ((None, BASE_CATEGORIES + ('nectar',)), ('Brood', STANDARD_BROOD_CATEGORIES),
                        ('Queen brood', QUEEN_BROOD_CATEGORIES))
BROOD_SHORT_LABELS = dict(zip(BROOD_CATEGORIES, (
    'Early', 'Early/Mid', 'Middle', 'Mid/Late', 'Late', 'Queen Mid', 'Queen Mid/Late', 'Queen Late')))
CATEGORY_LABELS = dict(zip(CATEGORIES, (
    'Bee', 'Hive', 'Chamber', 'Pollen',
    'Early brood: eggs',
    'Uncertain: eggs to larvae',
    'Middle brood: larvae',
    'Uncertain: larvae to pupae',
    'Late brood: pupae',
    'Queen brood: larvae',
    'Queen brood uncertain: larvae to pupae',
    'Queen brood: pupae',
    'Nectar source',
)))
BROOD_DESCRIPTIONS = dict(zip(BROOD_CATEGORIES, (
    'Eggs', 'Late-stage eggs to early-stage larvae', 'Larvae',
    'Late-stage larvae to early-stage pupae', 'Pupae',
    'Visually identified queen brood: larvae',
    'Visually identified queen brood: late-stage larvae to early-stage pupae',
    'Visually identified queen brood: pupae',
)))
CATEGORY_COLORS = {
    'chamber': (255, 0, 0), 'hive': (255, 255, 0), 'pollen': (255, 165, 0),
    'brood_early': (80, 205, 235), 'brood_early_middle': (90, 170, 125),
    'brood_middle': (65, 190, 80), 'brood_middle_late': (195, 130, 190),
    'brood_late': (210, 85, 165),
    'queen_brood_middle': (35, 120, 245),
    'queen_brood_middle_late': (155, 90, 235),
    'queen_brood_late': (245, 80, 100),
    'nectar': (0, 180, 160),
}
# Preserve the old compositing order. The combined image is only a display view.
MASK_CATEGORIES = ('bee', 'chamber', 'hive', 'pollen', 'nectar') + BROOD_CATEGORIES
MASK_ATTRIBUTES = tuple((category, f'{category}_mask') for category in MASK_CATEGORIES)

# Keep existing saved-map values, including unresolved=7, unchanged.
BROOD_UNRESOLVED_LABEL = 7
BROOD_MAP_LABELS = dict(zip(BROOD_CATEGORIES, (2, 3, 4, 5, 6, 8, 9, 10)))


def category_label(category):
    return CATEGORY_LABELS.get(category, category)


def training_categories(model_type):
    if model_type == 'brood':
        return BROOD_CATEGORIES
    if model_type in CATEGORIES:
        return (model_type,)
    raise ValueError(f'Unknown model_type {model_type!r}')
