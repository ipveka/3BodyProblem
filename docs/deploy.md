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

## Vercel (all-in-one, serverless)

Vercel hosts the whole app as one project: the static React build on the CDN and
the FastAPI backend as a Python serverless function. Suitable because the API is
stateless (request → simulate → response). Configuration is in `vercel.json`.

| Piece | How |
|-------|-----|
| Frontend | `buildCommand: cd frontend && npm install && npm run build`, `outputDirectory: frontend/dist` |
| Backend | `api/index.py` exposes the ASGI `app`; `includeFiles` bundles `backend/` + `core/` |
| Routing | `/api/*` and `/health` rewrite to the function; everything else is the static site |
| Deps | root `requirements.txt` (fastapi + numpy only; Vercel detects Python from it) |

### Steps

1. Push to GitHub.
2. Vercel: **Add New → Project** → import the repo → **Deploy** (defaults from
   `vercel.json`).

### Constraints & design choices

- **SciPy is not bundled.** It is imported lazily in the engine, so
  `rk4`/`euler`/`verlet` work with numpy alone; the optional `"scipy"` method is
  unavailable on Vercel. This keeps the function under the serverless size limit.
- **Same-origin, no env var.** The frontend defaults to a relative API base in
  production, and the function is served on the same domain — so no CORS and no
  `VITE_API_URL` needed.
- **Limits (Hobby/free):** `maxDuration` is set to 60s; serverless responses are
  capped (~4.5 MB). Large trajectories (high body count × many steps) can exceed
  these — keep `n_points` modest or downsample. Hobby is free for personal use;
  commercial use needs Pro.

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
