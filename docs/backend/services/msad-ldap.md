# `services/msad_ldap.py` - MS AD/LDAP Kommentar

Hinweis`backend/app/services/msad_ldap.py` Hinweis

## Kommentar
*   **LDAP Kommentar**: Kommentar
*   **BenutzerKommentar**: Kommentar
*   **BenutzerKommentar**: Kommentar
*   **BenutzerKommentar**: Kommentar

## Kommentar
Hinweis`ldap3` Hinweis

Hinweis
```python
from ldap3 import Server, Connection, ALL, SUBTREE
from backend.app.core.config import settings
import logging

logger = logging.getLogger(__name__)

def authenticate_ldap_user(username: str, password: str) -> bool:
    server = Server(settings.LDAP_SERVER, port=settings.LDAP_PORT, use_ssl=settings.LDAP_USE_SSL, get_info=ALL)
    conn = Connection(server, user=f"uid={username},{settings.LDAP_BASE_DN}", password=password, auto_bind=True)
    if conn.bind():
        logger.info(f"LDAP authentication successful for user: {username}")
        conn.unbind()
        return True
    else:
        logger.warning(f"LDAP authentication failed for user: {username}, error: {conn.result}")
        conn.unbind()
        return False

def get_ldap_user_info(username: str) -> Optional[dict]:
    server = Server(settings.LDAP_SERVER, port=settings.LDAP_PORT, use_ssl=settings.LDAP_USE_SSL, get_info=ALL)
    conn = Connection(server, user=settings.LDAP_BIND_DN, password=settings.LDAP_BIND_PASSWORD, auto_bind=True)
    if conn.bind():
        conn.search(
            search_base=settings.LDAP_BASE_DN,
            search_filter=f'(uid={username})',
            search_scope=SUBTREE,
            attributes=['mail', 'cn', 'memberOf'] # Hinweis
        )
        if conn.entries:
            user_entry = conn.entries[0]
            user_info = {
                "username": str(user_entry.cn),
                "email": str(user_entry.mail) if hasattr(user_entry, 'mail') else None,
                "groups": [str(g) for g in user_entry.memberOf] if hasattr(user_entry, 'memberOf') else []
            }
            conn.unbind()
            return user_info
        conn.unbind()
    return None
```

## Kommentar
`/backend/app/services/msad_ldap.py`