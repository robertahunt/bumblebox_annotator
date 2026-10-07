"""Consistent category groups for creating and reclassifying instances."""

from core.categories import BROOD_DESCRIPTIONS, CATEGORY_MENU_GROUPS, category_label


def add_category_menu_actions(menu, current_category=None):
    actions = {}
    for title, categories in CATEGORY_MENU_GROUPS:
        target = menu.addMenu(title) if title else menu
        for category in categories:
            label = category_label(category)
            if category == current_category:
                label += ' (current)'
            action = target.addAction(label)
            action.setData(category)
            action.setEnabled(category != current_category)
            if category in BROOD_DESCRIPTIONS:
                action.setToolTip(BROOD_DESCRIPTIONS[category])
            actions[category] = action
    return actions
