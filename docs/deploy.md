# Deployment Guide

The app deploys as two pieces: a **FastAPI backend** (the engine over HTTP) and a
**static React frontend**. This guide covers local Docker and Render.

## Files involved

| File | Purpose |
|------|---------|
| `Dockerfile` | Lean backend image (engine + API only) via `requirements-api.txt` |
| `.dockerignore` | Keeps the Docker build context small |
| `docker-compose.yml` | One-command local bring-up (backend + frontend) |
| `render.yaml` | Render Blueprint provisioning both services |
| `requirements-api.txt` | Backend-only runtime deps (numpy, scipy, fastapi, uvicorn) |

## Local (Docker Compose)

```bash
docker compose up --build
# Backend  -> http://localhost:8000  (docs at /docs)
# Frontend -> http://localhost:5173
```

## Render (Blueprint)

The repository includes a `render.yaml` Blueprint, so you don't configure
services by hand.

1. Push the repository to GitHub.
2. In Render: **New → Blueprint** and select the repo.
3. Render provisions:
   - **`3body-api`** — a Python web service.
     - Build: `pip install -r requirements-api.txt`
     - Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
     - Health check: `/health`
   - **`3body-frontend`** — a static site.
     - Build: `npm ci && npm run build` (root dir `frontend`)
     - Publish: `frontend/dist`, with an SPA rewrite to `index.html`

### Cross-service URLs

The two services reference each other through environment variables defined in
`render.yaml`:

| Variable | Service | Meaning |
|----------|---------|---------|
| `VITE_API_URL` | frontend (build-time) | Base URL of the backend |
| `ALLOWED_ORIGINS` | backend (CORS) | Allowed frontend origin |

They default to `https://3body-api.onrender.com` and
`https://3body-frontend.onrender.com`. If Render assigns different URLs (e.g. on
a name collision), update these two values in `render.yaml` or the dashboard and
redeploy. Because Vite inlines `VITE_API_URL` at build time, the frontend must be
rebuilt after changing it.

## Other platforms

The backend is a standard ASGI app, so any platform that can run a container or a
`uvicorn` process works (Railway, Fly.io, a VM, ...). Build from the `Dockerfile`
or replicate its steps: install `requirements-api.txt`, then run
`uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. Host the built
`frontend/dist` on any static host (Cloudflare Pages, Netlify, S3, ...).

## Post-deployment checklist

- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `GET /api/presets` lists the example systems
- [ ] The frontend loads and a preset simulation runs and animates
- [ ] CORS: the frontend origin is in `ALLOWED_ORIGINS`

## Notes

- Render's free tier spins services down after inactivity; the first request
  after idle incurs a cold start. The `/health` check keeps deploys honest.
- There's no authentication by default. Add it (and tighten `ALLOWED_ORIGINS`)
  before exposing a public, write-heavy instance.
