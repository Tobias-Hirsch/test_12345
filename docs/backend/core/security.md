# `core/security.py` - Kommentar

Hinweis`backend/app/core/security.py` Hinweis

## Kommentar
*   **PasswortKommentar**: Kommentar
*   **JWT Kommentar**: Kommentar
*   **JWT Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
1.  **PasswortHinweis**: Hinweis`passlib` Hinweis`bcrypt` Oder `PBKDF2`)Hinweis
    ```python
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)
    ```
2.  **JWT Hinweis**: Hinweis`python-jose` Hinweis`SECRET_KEY` Hinweis`ALGORITHM` Hinweis
    ```python
    from datetime import datetime, timedelta
    from typing import Optional
    from jose import jwt
    from backend.app.core.config import settings

    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    ```
3.  **JWT Hinweis**: Hinweis
    ```python
    from jose import JWTError, jwt
    from fastapi import HTTPException, status

    def decode_access_token(token: str) -> dict:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    ```

## Kommentar
`/backend/app/core/security.py`