# Public deployment reverse proxy

This folder contains the reverse-proxy configuration that should be applied when the app is exposed to the public internet.

## Nginx body-size limit

Use [`nginx-research-agent.conf`](./nginx-research-agent.conf) as the starting point for the server Nginx site config.

The important public upload limit is:

```nginx
client_max_body_size 30m;
```

It appears both at the `server` level and on the upload endpoint:

```nginx
location = /api/literature-analysis/pdf {
    client_max_body_size 30m;
    limit_req zone=research_agent_upload burst=10 nodelay;
    proxy_pass http://research_agent_app;
}
```

Keep this value aligned with the app setting:

```env
MAX_UPLOAD_TOTAL_MB=30
```

## Deployment checklist

1. Use Python 3.12.13 for the API and worker processes.
2. Start the backing services, including ClamAV:

   ```bash
   docker compose -f docker-compose.persistence.yml up -d
   ```

3. Set upload safety variables in `.env`:

   ```env
   MAX_UPLOAD_TOTAL_MB=30
   MAX_UPLOAD_FILE_MB=15
   MAX_UPLOAD_FILES=4
   PDF_UPLOAD_MAX_PAGE_COUNT=300
   MAX_UPLOAD_EXTRACTED_TEXT_CHARS=300000
   UPLOAD_PARSE_TIMEOUT_SECONDS=30
   PDF_PARSER_SANDBOX=enabled
   MAX_DOCX_UNZIPPED_MB=20
   MAX_DOCX_XML_MB=10
   MAX_DOCX_ZIP_ENTRIES=200
   MAX_DOCX_COMPRESSION_RATIO=100
   UPLOAD_VIRUS_SCAN=required
   CLAMAV_HOST=127.0.0.1
   CLAMAV_PORT=3310
   CLAMAV_TIMEOUT_SECONDS=10
   ```

4. Copy `deploy/nginx-research-agent.conf` into the server's Nginx sites directory.
5. Replace `server_name example.com;` with the real domain.
6. Confirm the app is listening on the configured upstream, default `127.0.0.1:8000`.
7. Enable the site and validate Nginx:

   ```bash
   nginx -t
   systemctl reload nginx
   ```

## Why this is needed

The app already enforces upload limits internally, but the reverse proxy rejects oversized requests before they reach Python. That protects memory, CPU, and request parsing capacity at the edge.
