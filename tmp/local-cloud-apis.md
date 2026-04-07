# Localized Cloud-Style APIs

These OpenAI-style endpoints now have local implementations instead of falling back to `501`:

- `GET /files`
- `POST /files`
- `GET /files/{file_id}`
- `GET /files/{file_id}/content`
- `DELETE /files/{file_id}`
- `GET /responses/{response_id}`
- `GET /responses/{response_id}/input_items`
- `DELETE /responses/{response_id}`
- `POST /realtime/client_secrets`
- `POST /realtime/transcription_sessions`

Implementation notes:
- Files are stored on disk under `.local_api_state/files/`
- Responses retrieval uses local in-memory persistence for the current server process
- Realtime secrets and transcription sessions return local ephemeral credentials and session objects
- Deepgram and LM Studio remain split into separate namespaces to avoid surface collisions
