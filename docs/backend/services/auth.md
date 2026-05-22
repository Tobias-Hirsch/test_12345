# `services/auth.py` - Kommentar

Hinweis`backend/app/services/auth.py` Hinweis

## Kommentar
*   **BenutzerKommentar**: Kommentar
*   **JWT Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **BerechtigungKommentar**: Kommentar

## Kommentar
1.  **`authenticate_user(db: Session, username: str, password: str)`**:
    *   Kommentar
    *   Kommentar`security.verify_password` Kommentar
    *   Kommentar
2.  **`create_access_token(data: dict, expires_delta: Optional[timedelta] = None)`**:
    *   Kommentar`security.create_access_token` Kommentar
3.  **`get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db))`**:
    *   Kommentar
    *   Kommentar`security.decode_access_token` Kommentar
    *   Kommentar
    *   Kommentar`HTTPException`. 
4.  **`get_current_active_user(current_user: User = Depends(get_current_user))`**:
    *   Kommentar`get_current_user` Kommentar
5.  **`get_current_active_superuser(current_user: User = Depends(get_current_active_user))`**:
    *   Kommentar`get_current_active_user` Kommentar

## Kommentar
`/backend/app/services/auth.py`