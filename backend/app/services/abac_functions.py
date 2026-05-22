from datetime import datetime
from typing import Any

def is_resource_owner(user_id: str, resource_owner_id: str) -> bool:
    """
    Hinweis
    """
    return str(user_id) == str(resource_owner_id)

def is_within_working_hours(current_time: datetime) -> bool:
    """
    Hinweis
    """
    return 9 <= current_time.hour < 17

# Kommentar