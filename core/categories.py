"""Stable annotation category keys and appearance-based brood labels."""

BASE_CATEGORIES = ('bee', 'hive', 'chamber', 'pollen')
BROOD_CATEGORIES = ('brood_early', 'brood_early_middle', 'brood_middle',
                    'brood_middle_late', 'brood_late')
CATEGORIES = BASE_CATEGORIES + BROOD_CATEGORIES
BROOD_SHORT_LABELS = dict(zip(BROOD_CATEGORIES, ('Early', 'Early/Mid', 'Middle', 'Mid/Late', 'Late')))
CATEGORY_LABELS = dict(zip(CATEGORIES, (
    'Bee', 'Hive', 'Chamber', 'Pollen',
    'Early brood: eggs',
    'Uncertain: eggs to larvae',
    'Middle brood: larvae',
    'Uncertain: larvae to pupae',
    'Late brood: pupae',
)))
BROOD_DESCRIPTIONS = dict(zip(BROOD_CATEGORIES, (
    'Eggs', 'Late-stage eggs to early-stage larvae', 'Larvae',
    'Late-stage larvae to early-stage pupae', 'Pupae',
)))
CATEGORY_COLORS = {
    'chamber': (255, 0, 0), 'hive': (255, 255, 0), 'pollen': (255, 165, 0),
    'brood_early': (80, 205, 235), 'brood_early_middle': (90, 170, 125),
    'brood_middle': (65, 190, 80), 'brood_middle_late': (195, 130, 190),
    'brood_late': (210, 85, 165),
}
# Preserve the old compositing order. The combined image is only a display view.
MASK_CATEGORIES = ('bee', 'chamber', 'hive', 'pollen') + BROOD_CATEGORIES
MASK_ATTRIBUTES = tuple((category, f'{category}_mask') for category in MASK_CATEGORIES)


def category_label(category):
    return CATEGORY_LABELS.get(category, category)


def training_categories(model_type):
    if model_type == 'brood':
        return BROOD_CATEGORIES
    if model_type in CATEGORIES:
        return (model_type,)
    raise ValueError(f'Unknown model_type {model_type!r}')
