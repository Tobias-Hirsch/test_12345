# `routers/files.py` - Kommentar

Hinweis`backend/app/routers/files.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentaröschen**: KommentaröschenKommentar

## Kommentar
1.  **`upload_file(file: UploadFile = File(...), current_user: User = Depends(auth.get_current_active_user), db: Session = Depends(get_db))`**:
    *   Kommentar
    *   Kommentar
    *   Kommentar
    *   Kommentar
    *   `@router.post("/uploadfile")`
2.  **`download_file(file_id: int, current_user: User = Depends(auth.get_current_active_user), db: Session = Depends(get_db))`**:
    *   Kommentar`file_id` Kommentar
    *   Kommentar
    *   `@router.get("/download/{file_id}")`
3.  **`list_files(current_user: User = Depends(auth.get_current_active_user), db: Session = Depends(get_db))`**:
    *   Kommentar
    *   Kommentar
    *   `@router.get("/list_files")`
4.  **`delete_file(file_id: int, current_user: User = Depends(auth.get_current_active_user), db: Session = Depends(get_db))`**:
    *   Kommentar`file_id` Löschen MinIO MittelKommentar
    *   `@router.delete("/delete_file/{file_id}")`

## Kommentar
`/backend/app/routers/files.py`