# `services/abac_functions.py` - ABAC Kommentar

Hinweis`backend/app/services/abac_functions.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **BedingungKommentar**: Kommentar

## Kommentar
Hinweisührt ausHinweis

Hinweis
```python
from typing import Any, List, Union

def equals(attr1: Any, attr2: Any) -> bool:
    """Checks if two attributes are equal."""
    return attr1 == attr2

def greater_than(attr1: Union[int, float], attr2: Union[int, float]) -> bool:
    """Checks if attr1 is greater than attr2."""
    return attr1 > attr2

def contains(collection: List[Any], item: Any) -> bool:
    """Checks if a collection contains an item."""
    return item in collection

def starts_with(text: str, prefix: str) -> bool:
    """Checks if a string starts with a prefix."""
    return text.startswith(prefix)

def is_member_of_group(user_groups: List[str], required_group: str) -> bool:
    """Checks if a user is a member of a specific group."""
    return required_group in user_groups

# Kommentar
```

## Kommentar
`/backend/app/services/abac_functions.py`