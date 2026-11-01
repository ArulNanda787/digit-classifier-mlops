# MNIST Digit Classifier — End-to-End MLOps on AWS

A handwritten-digit CNN taken from raw data to a monitored production deployment:
MLflow-tracked training → FastAPI service → Docker image → GitHub Actions CI/CD →
EC2 provisioned with Terraform → Prometheus + Grafana monitoring.

## Run it (Windows, macOS, Linux)

Requires Docker Desktop (on Windows: WSL 2 backend, Linux containers).

```
docker run -d --name digit-api -p 8000:8000 ghcr.io/arulnanda787/digit-classifier:latest
```

Open http://localhost:8000 and draw a digit · API docs: http://localhost:8000/docs ·
Stop: `docker rm -f digit-api`

The image is built for both `linux/amd64` and `linux/arm64`.

## Architecture

```
Laptop                       GitHub Actions                         AWS ap-south-1 (Terraform)
ETL → train 2 models  ──git push──►  tests ─► build + smoke test ─► deploy ──► ECR ──pull──► EC2 :80 (FastAPI)
      MLflow picks best                          │                                         │ /metrics
      models/digit_cnn.keras                     └─► GHCR (public)              Prometheus :9090 ─► Grafana :3000
```

## Project layout

| Path | What it is |
|---|---|
| `src/data_preprocessing.py` | ETL: download MNIST, scale to [0, 1], save `data/processed/mnist.npz` |
| `src/train.py` | Trains a CNN and a deliberately weak MLP as MLflow runs; saves the best to `models/digit_cnn.keras` |
| `app/` | FastAPI: `/` canvas, `/health`, `/predict`, `/metrics`; preprocessing; model loading |
| `tests/` | pytest: preprocessing, API (fake model), metrics |
| `Dockerfile`, `requirements-serve.txt` | Serving image (no MLflow), non-root, health check |
| `.github/workflows/ci.yml` | test → build/smoke-test/publish to GHCR → deploy to AWS |
| `infra/` | Terraform: ECR, security group, IAM role, EC2 + boot script |
| `monitoring/` | Prometheus config, Grafana datasource and dashboard, docker-compose |
| `scripts/` | Sample export, accuracy check, load test, AWS and monitoring setup |

## Phase 1 — Train and compare models

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_preprocessing
python -m src.train
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

In MLflow, select `cnn-baseline` and `weak-mlp` → **Compare**. The run tagged
`selected_for_deployment=true` is the model in `models/digit_cnn.keras`
(chosen on validation accuracy).

## Phase 2 — Run the app locally

```bash
python -m pytest -v
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

python scripts/export_samples.py --count 20
python scripts/predict_samples.py
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Drawing canvas |
| GET | `/health` | `{"status": "ok", "model_loaded": true}` |
| POST | `/predict` | Image upload (`file`) → `digit`, `confidence`, `probabilities` |
| GET | `/metrics` | Prometheus metrics |

Errors: 400 unreadable or blank image · 413 over 5 MB · 415 not an image · 422 no file · 503 model not loaded.

Use `python -m uvicorn` (not bare `uvicorn`) so the project's venv and Keras version are used.

## Phase 3 — Docker

```bash
docker build -t digit-classifier:1.0 .
docker run -d --name digit-api -p 8000:8000 digit-classifier:1.0
docker ps                                    # STATUS becomes (healthy)

docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/arulnanda787/digit-classifier:latest --push .
```

## Phase 4 — CI/CD

Every push to `main` runs three jobs in order: **Unit tests** → **Build, smoke-test,
publish** (GHCR, tagged `sha-<commit>` and `latest`) → **Deploy to AWS** (ECR push,
redeploy via SSM, public health check). Pull requests are tested only.

Repository settings needed (created by `scripts/setup_github_deploy.sh`):
secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`; variables `ECR_REPOSITORY`,
`EC2_INSTANCE_ID`, `EC2_PUBLIC_IP`. The GHCR package must grant this repository
**Write** under *Manage Actions access*.

## Phase 5 — AWS with Terraform

```bash
bash scripts/aws_deploy.sh          # tfvars, key pair, terraform plan/apply, image to ECR, first deploy
bash scripts/setup_github_deploy.sh # AWS deploy user + GitHub secrets/variables
```

Or by hand: `cd infra && terraform init && terraform plan && terraform apply`.
The app is served on `http://<public_ip>` (`terraform output`).

## Phase 6 — Monitoring

```bash
bash scripts/deploy_monitoring.sh
python scripts/load_test.py --url http://<public_ip> --requests 300
```

- Grafana: `http://<public_ip>:3000` (admin / mlops-demo) — dashboard *Digit Classifier — Live*
- Prometheus: `http://<public_ip>:9090/targets`

Ports 22, 3000 and 9090 are open only to the IP in `infra/terraform.tfvars`.

## Tear down

```bash
cd infra && terraform destroy
```

Removes every AWS resource. The public GHCR image keeps working.
