# Amazon Sentiment Analysis Pipeline and MLOps Platform

An enterprise-grade, cloud-native machine learning operations (MLOps) platform and sentiment analysis pipeline built using AWS services with automated CI/CD via GitHub Actions.

The platform orchestrates distributed data ingestion and ETL with AWS Glue (PySpark), model fine-tuning with AWS SageMaker (DistilBERT) and MLflow, serverless batch inference with AWS ECS Fargate, and full-stack web application delivery with FastAPI (secured by AWS Cognito) and Streamlit.

---

## Architecture Documentation

| Document | File Path | Focus Area |
|---|---|---|
| High-Level Design (HLD) | [HLD.md](HLD.md) or [docs/HLD.md](docs/HLD.md) | System goals, cloud topology, subsystem decomposition, security, and trade-offs |
| Low-Level Design (LLD) | [LLD.md](LLD.md) or [docs/LLD.md](docs/LLD.md) | Class/function signatures, Pydantic schemas, Spark algorithms, and test specs |
| Knowledge Graph Report | [GRAPH_REPORT.md](GRAPH_REPORT.md) | Graph-theoretic metrics, 6 Leiden communities, God nodes, and coupling audit |
| Interactive Graph Explorer | [graphify-out/graph.html](graphify-out/graph.html) | Standalone D3.js force-directed knowledge graph visualization |

---

## System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                      CLIENT / USER INTERFACE                                       |
|                                                                                                    |
|    +-----------------------------+                           +--------------------------------+    |
|    |  Streamlit UI (Port 8501)   |                           |    API Consumers / Swagger     |    |
|    |  Login / Upload / Dashboard |                           |    http://localhost:8000/docs  |    |
|    +--------------+--------------+                           +---------------+----------------+    |
+-------------------|----------------------------------------------------------|---------------------+
                    |                                                          |
                    | HTTP POST (Auth, Trigger, Summary)                       | HTTP REST
                    v                                                          v
+----------------------------------------------------------------------------------------------------+
|                                    API GATEWAY & AUTHENTICATION                                    |
|                                                                                                    |
|    +------------------------------------------------------------------------------------------+    |
|    | FastAPI Backend Application (Port 8000)                                                  |    |
|    | Routes: /login, /trigger_pipeline, /generate_dashboard, /health, /                       |    |
|    +--------------------------------------------+---------------------------------------------+    |
|                                                 |                                                  |
|                        +------------------------v------------------------+                         |
|                        | AWS Cognito User Pool                           |                         |
|                        | ID / Access Token Issuance & Verification       |                         |
|                        +-------------------------------------------------+                         |
+-------------------------------------------------|--------------------------------------------------+
                                                  |
            +-------------------------------------+-------------------------------------+
            | AWS ECS run_task (Async Fargate Task)                                     | S3 Status & Summary
            v                                                                           v
+-------------------------------------------+               +----------------------------------------+
|       BATCH INFERENCE SUBSYSTEM           |               |           AWS S3 APP BUCKET            |
|                                           |               |        (mlops-sentiment-app)           |
|  +-------------------------------------+  |               |                                        |
|  | AWS ECS Fargate Task                |  |  Read Raw     |  - uploads/raw/{user}/data.csv         |
|  | Container: sentiment-pipeline:latest|--+-------------->|  - processed/{user}/processed.csv      |
|  |                                     |  |               |  - served/{user}/served.csv            |
|  | 1. run_all.py (Coordinator)         |  |  Write Served |  - pipeline_status/status.json         |
|  | 2. run_clean.py (PySpark Cleansing) |<-+---------------+  - models/model.tar.gz                 |
|  | 3. run_infer.py (DistilBERT Infer)  |  |               |  - dashboard/summary.json              |
|  +-------------------------------------+  |               +----------------------------------------+
+-------------------------------------------+                                   ^
                                                                                |
                                                           Model Promotion Script
                                                           app_bucket_s3.py
                                                                                |
+-------------------------------------------------------------------------------+--------------------+
|                                     DATA ENGINEERING & TRAINING                                    |
|                                                                                                    |
|  +------------------------------------+                    +------------------------------------+  |
|  | AWS Glue Data Pipeline (PySpark)   |                    | AWS SageMaker & MLflow             |  |
|  |                                    |                    |                                    |  |
|  | 1. glue_data_filtering.py          |                    | 1. trigger.py (Estimator Launcher) |  |
|  | 2. glue_data_preprocessing.py      |                    | 2. train.py (GPU Fine-Tuning)      |  |
|  | 3. glue_schema_validation.py       |                    | 3. test.py (20k Sample Eval)       |  |
|  | 4. glue_anomaly_detection.py       |                    | 4. evaluate.py (Metric Generation) |  |
|  | 5. glue_sampling.py                |                    | 5. deploy.py (Endpoint Serving)    |  |
|  +-----------------+------------------+                    +-----------------+------------------+  |
|                    |                                                         |                      |
|                    | Read/Write Partitions                                   | Read Data & Save Tar |
|                    v                                                         v                      |
|  +-----------------------------------------------------------------------------------------------+  |
|  | AWS S3 DATA LAKE (mlops-sentiment-analysis-data)                                              |  |
|  | - raw/reviews.csv | bronze/clean/ | silver/sampled/ | models/model.tar.gz | mlflow/           |  |
|  +-----------------------------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
```

---

## Key Features

- Distributed Data Processing: PySpark jobs running on AWS Glue execute text normalization, schema validation, anomaly removal, and stratified sampling.
- Transformer Classification: DistilBERT fine-tuned sequence classification model reaching 93.14% accuracy and 93.05% F1 score.
- Serverless Batch Inference: Isolated AWS ECS Fargate container running automated cleansing and model evaluation on user uploads.
- Secure API Gateway: FastAPI backend with OAuth2/JWT token verification backed by AWS Cognito.
- Real-time Analytics Dashboard: Multi-page Streamlit application providing interactive visualization of sentiment ratios and review lengths.
- Offline Simulation Mode: Built-in mock authentication and task simulation enabling complete local development without AWS credentials.
- Knowledge Graph Navigation: Graphify integration mapping 31 structural nodes and 48 directed operational edges.
- Production Automated Tests: 13 unit and integration tests executing in under 2 seconds.

---

## Project Structure

```
Amazon-Sentiment-Analysis-Pipeline-main/
├── README.md
├── HLD.md
├── LLD.md
├── GRAPH_REPORT.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/
│   ├── HLD.md
│   └── LLD.md
├── graphify-out/
│   ├── graph.json
│   ├── graph.html
│   └── GRAPH_REPORT.md
├── .github/workflows/
│   └── ci_cd.yml
├── scripts/
│   ├── glue/
│   │   ├── glue_data_filtering.py
│   │   ├── glue_data_preprocessing.py
│   │   ├── glue_schema_validation.py
│   │   ├── glue_anomaly_detection.py
│   │   └── glue_sampling.py
│   ├── s3/
│   │   ├── app_bucket_s3.py
│   │   ├── s3_move_to_default.py
│   │   └── upload_to_s3.py
│   └── sage_maker/
│       ├── trigger.py
│       ├── train.py
│       ├── test.py
│       ├── deploy.py
│       ├── evaluate.py
│       ├── inference.py
│       ├── config_model.json
│       └── model_evaluation_summary.json
├── sentiment-app/
│   ├── requirements.txt
│   ├── restartapp.bat
│   ├── backend/
│   │   ├── .env.example
│   │   ├── auth.py
│   │   ├── main.py
│   │   └── utils.py
│   ├── frontend/
│   │   ├── streamlit_app.py
│   │   ├── test.py
│   │   └── pages/
│   │       ├── upload_and_trigger.py
│   │       └── dashboard.py
│   ├── sentiment-pipeline/
│   │   ├── Dockerfile
│   │   ├── run_all.py
│   │   ├── run_clean.py
│   │   └── run_infer.py
│   └── deployment/
│       ├── ec2_start_and_deploy.py
│       └── ec2_stop.py
└── tests/
    ├── test_api.py
    ├── test_glue_logic.py
    └── test_inference_logic.py
```

---

## Quickstart Guide

### Option A: Local Development (Offline Mock Mode)

You can launch and test the full web stack locally without AWS credentials or cloud infrastructure.

#### 1. Clone and Install Dependencies
```bash
git clone https://github.com/AshutoshYadav/Amazon-Sentiment-Analysis-Pipeline.git
cd Amazon-Sentiment-Analysis-Pipeline
pip install -r requirements.txt
pip install -r sentiment-app/requirements.txt
```

#### 2. Configure Environment Variables
Copy `.env.example` into `sentiment-app/backend/.env`:
```bash
cp .env.example sentiment-app/backend/.env
```
Ensure the mock flags are active:
```ini
MOCK_AUTH=true
MOCK_PIPELINE=true
APP_BUCKET=mlops-sentiment-app
AWS_REGION=us-east-1
```

#### 3. Start the Backend API (FastAPI)
```bash
cd sentiment-app
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The API is now live at `http://localhost:8000`. View interactive Swagger docs at `http://localhost:8000/docs`.

#### 4. Start the Frontend UI (Streamlit)
In a separate terminal:
```bash
cd sentiment-app/frontend
streamlit run streamlit_app.py --server.port 8501
```
Open `http://localhost:8501` in your browser:
- Log in with any username and password (e.g., `admin` / `password123`).
- Navigate to "Upload and Trigger" to submit review CSV files.
- View real-time sentiment distribution in the "Dashboard".

---

### Option B: Cloud Deployment (Production AWS Mode)

#### 1. Prerequisites
- An active AWS Account.
- AWS CLI configured with administrator or deployment credentials: `aws configure`.
- Two Amazon S3 Buckets created:
  - `mlops-sentiment-analysis-data` (ETL and model registry)
  - `mlops-sentiment-app` (application uploads and serving)
- AWS Cognito User Pool with an App Client (Client Secret not required).
- AWS ECS Cluster with Fargate capacity provider.

#### 2. Configure Production `.env`
Update `sentiment-app/backend/.env`:
```ini
MOCK_AUTH=false
MOCK_PIPELINE=false
AWS_REGION=us-east-1
COGNITO_USER_POOL_ID=us-east-1_xxxxxxxxx
COGNITO_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxx
APP_BUCKET=mlops-sentiment-app
DATA_BUCKET=mlops-sentiment-analysis-data
ECS_CLUSTER=sentiment-analysis-cluster
ECS_TASK_DEFINITION=sentiment-pipeline-task:1
ECS_SUBNETS=subnet-0123456789abcdef0,subnet-0abcdef0123456789
ECS_SECURITY_GROUPS=sg-0123456789abcdef0
```

#### 3. Build and Publish Inference Container
```bash
cd sentiment-app/sentiment-pipeline
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker build -t sentiment-pipeline .
docker tag sentiment-pipeline:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/sentiment-pipeline:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/sentiment-pipeline:latest
```

---

## Data Pipeline (AWS Glue)

The offline data pipeline transforms raw, noisy review dumps into high-quality training sets:

| Step | Script | Input | Output | Purpose |
|---|---|---|---|---|
| 1 | `glue_data_filtering.py` | `raw/reviews.csv` | `splits/train/`, `splits/test/`, `splits/serve/` | Deterministic 80/10/10 split on customer ID |
| 2 | `glue_data_preprocessing.py` | `splits/train/` | `bronze/clean/train/` | Whitespace normalization, HTML tag stripping, lowercasing |
| 3 | `glue_schema_validation.py` | `bronze/clean/train/` | `validated/train/` | Type checking, null validation, length profiling metadata |
| 4 | `glue_anomaly_detection.py` | `validated/train/` | `anomaly_filtered/` | Clamps star ratings to [1, 5] and filters length outliers |
| 5 | `glue_sampling.py` | `anomaly_filtered/` | `silver/sampled/train_data_sampled.csv` | Stratified balancing to 50,000 binary records (1-2 vs 4-5 stars) |

To run the Glue pipeline manually:
```bash
python scripts/glue/glue_data_filtering.py
python scripts/glue/glue_data_preprocessing.py
python scripts/glue/glue_schema_validation.py
python scripts/glue/glue_anomaly_detection.py
python scripts/glue/glue_sampling.py
```

---

## Model Training and Evaluation (AWS SageMaker)

### 1. Launch Fine-Tuning Job
Fine-tune DistilBERT on GPU compute (`ml.g4dn.xlarge`):
```bash
python scripts/sage_maker/trigger.py
```
This triggers `scripts/sage_maker/train.py` inside SageMaker, tracking loss and metrics in MLflow.

### 2. Promote Model Artifacts
After training completes, promote the latest model archive to the app bucket:
```bash
python scripts/s3/app_bucket_s3.py
```

### 3. Evaluate Model Performance
```bash
python scripts/sage_maker/evaluate.py
```

#### Evaluation Results
| Metric | Value |
|---|---|
| Accuracy | 93.14% |
| Precision | 92.80% |
| Recall | 93.30% |
| F1 Score | 93.05% |

---

## API Reference

The backend exposes a REST API running on port 8000:

| Route | Method | Authorization | Description |
|---|---|---|---|
| `/` | GET | None | Service index verification |
| `/health` | GET | None | Healthcheck probe |
| `/login` | POST | None | Authenticate user against Cognito and retrieve JWT |
| `/trigger_pipeline` | POST | Bearer Token | Launch on-demand ECS Fargate batch inference task |
| `/generate_dashboard` | POST | Bearer Token | Parse served results and emit summary metrics |

### Example Trigger Request
```bash
curl -X POST http://localhost:8000/trigger_pipeline \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "usr_101", "file_name": "reviews.csv"}'
```

---

## Automated Verification Suite

Run the automated test suite using PyTest:
```bash
python -m pytest tests/ -v
```

### Test Coverage Breakdown
- `tests/test_api.py`: Validates API status, healthcheck, login flows, token enforcement, and protected route access control.
- `tests/test_glue_logic.py`: Validates text normalization, star rating bounds clamping, binary sentiment labeling, and short review filtering.
- `tests/test_inference_logic.py`: Validates class mapping and summary analytics calculation.

**Result**: 13 passed in under 2 seconds with zero external dependencies.

---

## CI/CD Pipeline

The GitHub Actions workflow is defined in `.github/workflows/ci_cd.yml`:
1. **Lint & Test**: Runs Flake8 syntax checks and executes the full PyTest suite on every pull request and branch push.
2. **ECR Image Build & Push**: Automatically builds the `sentiment-pipeline` Docker container and deploys it to Amazon ECR upon merge to `main`.

---

## Maintainer

**Ashutosh Yadav**  
Enterprise MLOps & Machine Learning Engineering
