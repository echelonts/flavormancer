# infra/

Dockerfiles, `docker-compose.yml`, and deployment for a single on-prem box.

- One Dockerfile per service (api, frontend, aroma-sidecar if used)
- Compose stack wiring the app, Postgres + pgvector, and the models
- Single-box deploy — Docker Compose, not Kubernetes (deliberate: small, known
  user count, on-prem simplicity is part of the product)
