# Deployment

## Docker Compose (PowerShell)

```powershell
Copy-Item .env.example .env            # set FLIGHTOPS_ADMIN_TOKEN
# put ai4i2020.csv in .\data\raw  (or generate the SYNTHETIC demo file, see README)
docker compose up --build -d
docker compose --profile train run --rm trainer      # one-shot training inside the stack
docker compose ps                                     # all services should be "healthy"
```

| Service | Port | Notes |
|---|---|---|
| mlflow | 5000 | SQLite backend + proxied artifacts on the `mlflow_data` volume |
| api | 8000 | `/docs`, `/health`, `/ready`, `/predict`, ... ; reads registry alias `champion` |
| dashboard | 8501 | Streamlit control center; also spawns training jobs |
| trainer | - | optional profile `train` |

Persistence: `mlflow_data` (experiments, registry, artifacts) and `state` (prediction log, governance audit, job state) are named volumes - `docker compose restart` / `down` + `up` keeps them (`down -v` deletes them). `./data` and `./reports` are bind-mounted.

Networking: services use the internal network (`http://mlflow:5000`, `http://api:8000`); only 5000/8000/8501 are published to the host.

## Promote a model

```powershell
docker compose exec dashboard python -m flightops_ml registry status
docker compose exec dashboard python -m flightops_ml registry approve --version 1 --user alice --reason "reviewed metrics"
docker compose exec dashboard python -m flightops_ml registry deploy  --version 1 --user alice
```
The API picks up the new champion within `serving.champion_refresh_seconds` (30 s) - no restart. Roll back with `registry rollback --user alice --reason "..."`.

## Security notes
* Set `FLIGHTOPS_API_KEY` to require `X-API-Key` on prediction endpoints; terminate TLS in front of the API for anything non-local.
* The dashboard admin panel is disabled unless `FLIGHTOPS_ADMIN_TOKEN` is set. Keep the dashboard on a trusted network.
* The MLflow server has no authentication; do not publish port 5000 beyond localhost.
* CI uses least-privilege permissions and an isolated MLflow store; secrets only via GitHub Secrets.
