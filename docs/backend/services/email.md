# `services/email.py` - Kommentar

Hinweis`backend/app/services/email.py` Hinweis

## Kommentar
*   **SMTP Kommentar**: Kommentar
*   **Kommentar**: Kommentar`smtplib` OderKommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: SendenKommentar
*   **Passwort zurücksetzenKommentar**: SendenKommentar
*   **Kommentar**: Kommentar

## Kommentar
1.  **SMTP Hinweis**: Hinweis`smtplib` Hinweis
    ```python
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from backend.app.core.config import settings
    import logging

    logger = logging.getLogger(__name__)

    async def send_email(
        to_email: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ):
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{settings.EMAILS_FROM_NAME} <{settings.EMAILS_FROM_EMAIL}>"
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                if settings.SMTP_TLS:
                    server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)
            logger.info(f"Email sent to {to_email} with subject: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            raise
    ```
2.  **`send_activation_email(email: str, username: str, token: str)`**:
    *   Kommentar
    *   Kommentar`send_email` SendenKommentar
3.  **`send_reset_password_email(email: str, username: str, token: str)`**:
    *   Kommentar
    *   Kommentar`send_email` SendenKommentar
4.  **`send_test_email(email: str)`**:
    *   SendenKommentar

## Kommentar
`/backend/app/services/email.py`