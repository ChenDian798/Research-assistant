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

1. Copy `deploy/nginx-research-agent.conf` into the server's Nginx sites directory.
2. Replace `server_name example.com;` with the real domain.
3. Confirm the app is listening on the configured upstream, default `127.0.0.1:8000`.
4. Enable the site and validate Nginx:

   ```bash
   nginx -t
   systemctl reload nginx
   ```

5. For public deployments, also enable virus scanning in `.env`:

   ```env
   UPLOAD_VIRUS_SCAN=required
   CLAMAV_HOST=127.0.0.1
   CLAMAV_PORT=3310
   ```

## Why this is needed

The app already enforces upload limits internally, but the reverse proxy rejects oversized requests before they reach Python. That protects memory, CPU, and request parsing capacity at the edge.
