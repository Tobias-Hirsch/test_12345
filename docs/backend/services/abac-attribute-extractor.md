# `services/abac_attribute_extractor.py` - ABAC Kommentar

Hinweis`backend/app/services/abac_attribute_extractor.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis

Hinweis
```python
from typing import Dict, Any
from fastapi import Request
from sqlalchemy.orm import Session
from backend.app.models.database import User # Hinweis
# from backend.app.models.database import Resource # Kommentar

def extract_abac_attributes(
    request: Request,
    current_user: User,
    db: Session,
    resource_id: Optional[int] = None,
    action: str = ""
) -> Dict[str, Any]:
    """
    Extracts attributes for ABAC policy evaluation.
    """
    attributes = {
        "subject": {
            "user_id": current_user.id,
            "username": current_user.username,
            "is_superuser": current_user.is_superuser,
            # ... Kommentar
        },
        "action": action,
        "resource": {},
        "environment": {
            "ip_address": request.client.host,
            "timestamp": datetime.utcnow().isoformat(),
            # ... Kommentar
        }
    }

    if resource_id:
        # Kommentar
        # resource = db.query(Resource).filter(Resource.id == resource_id).first()
        # if resource:
        #     attributes["resource"] = resource.to_dict() # Kommentar
        pass # HinweisägeHinweis

    # Kommentar
    # if request.method == "POST" or request.method == "PUT":
    #     request_body = await request.json()
    #     attributes["request_body"] = request_body

    return attributes
```

## Kommentar
`/backend/app/services/abac_attribute_extractor.py`