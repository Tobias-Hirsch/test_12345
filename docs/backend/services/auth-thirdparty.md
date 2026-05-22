# `services/auth_thirdparty.py` - Kommentar

Hinweis`backend/app/services/auth_thirdparty.py` Hinweis

## Kommentar
*   **OAuth Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **BenutzerKommentar**: Kommentar
*   **BenutzerRegistrieren/Anmelden**: Kommentar

## Kommentar
Hinweis`Authlib` OderHinweis

Hinweis
1.  **Hinweis**:
    ```python
    from authlib.integrations.starlette_client import OAuth
    from backend.app.core.config import settings

    oauth = OAuth()
    oauth.register(
        name='google',
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        authorize_url='https://accounts.google.com/o/oauth2/auth',
        access_token_url='https://accounts.google.com/o/oauth2/token',
        api_base_url='https://www.googleapis.com/oauth2/v1/',
        client_kwargs={'scope': 'openid email profile'},
    )
    ```
2.  **Hinweis**:
    ```python
    from starlette.requests import Request
    from starlette.responses import RedirectResponse

    async def login_google(request: Request):
        redirect_uri = request.url_for('auth_google')
        return await oauth.google.authorize_redirect(request, redirect_uri)
    ```
3.  **Hinweis**:
    ```python
    async def auth_google(request: Request, db: Session = Depends(get_db)):
        token = await oauth.google.authorize_access_token(request)
        user_info = await oauth.google.parse_id_token(token)
        # Kommentar
        # Kommentar
    ```

## Kommentar
`/backend/app/services/auth_thirdparty.py`