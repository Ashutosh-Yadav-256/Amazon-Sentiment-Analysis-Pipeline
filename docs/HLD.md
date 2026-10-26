# High-Level Design (HLD) Document

Project: Amazon Sentiment Analysis Pipeline and MLOps Platform
Maintainer: Ashutosh Yadav
Version: 2.4.0
Status: Approved

---

## 1. Executive Overview

The Amazon Sentiment Analysis Pipeline is an enterprise-grade, cloud-native machine learning operations (MLOps) platform. The system ingests large-scale unstructured customer product reviews, executes distributed data cleansing and validation using Apache Spark on AWS Glue, fine-tunes transformer-based language models (DistilBERT) with AWS SageMaker, packages containerized batch inference pipelines via AWS ECS Fargate, and serves end users through a decoupled web application comprising a FastAPI backend and a Streamlit frontend secured by AWS Cognito.

### 1.1 Objectives
- Ingestion Scalability: Process millions of Amazon customer reviews across product categories.
- Data Quality & Governance: Provide automated schema enforcement, whitespace normalization, outlier detection, and stratified class balancing.
- Model Performance: Achieve over 90% classification accuracy and F1 score on binary sentiment prediction (0: Negative, 1: Positive).
- Operational Decoupling: Isolate training, batch inference, and user-facing web services to eliminate resource contention and minimize idle cloud compute expenses.
- Production Security: Enforce OAuth2/JWT Bearer token verification via AWS Cognito across all sensitive API routes.

---

## 2. High-Level System Architecture

The platform follows a microservices and event-driven architecture centered around decoupled compute tiers and object storage data lakes.

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

## 3. Subsystem Decomposition

### 3.1 Data Ingestion and Lakehouse Architecture
The storage layout is segregated into two independent AWS S3 buckets to enforce a separation between data engineering workflows and client application state:

1. Data Lake Bucket (`s3://mlops-sentiment-analysis-data`):
   - Raw Zone: Ingested source files (`raw/reviews.csv`).
   - Bronze Zone: Cleaned, schema-validated, and deduplicated records (`bronze/clean/`).
   - Silver Zone: Stratified class-balanced datasets (`silver/sampled/train_data_sampled.csv`).
   - Gold Zone / Model Registry: Versioned PyTorch model archives (`models/{timestamp}/model.tar.gz`).
   - Tracking Zone: MLflow run artifacts and experiment telemetry (`mlflow/`).

2. Application Serving Bucket (`s3://mlops-sentiment-app`):
   - Client Uploads: Ingested client CSV files partitioned by user ID (`uploads/raw/{user_id}/`).
   - Processed Tier: Standardized text ready for inference (`processed/{user_id}/processed.csv`).
   - Inference Tier: Final classifications with prediction scores (`served/{user_id}/served.csv`).
   - State Tracking: Real-time batch task heartbeat and progress state (`pipeline_status/status.json`).
   - Active Production Weights: Promoted model archive used by containers (`models/model.tar.gz`).
   - Metric Cache: Aggregated distribution statistics for dashboard visualization (`dashboard/summary.json`).

### 3.2 Data Engineering Subsystem (AWS Glue)
The offline data engineering subsystem runs 5 modular PySpark jobs orchestrated sequentially:

```
[Raw Data] 
    |
    v
+---------------------------+
| 1. glue_data_filtering    | -> Split 80% Train, 10% Test, 10% Serve
+---------------------------+
    |
    v
+---------------------------+
| 2. glue_data_preprocessing| -> Whitespace normalization, HTML tag removal, lowercasing
+---------------------------+
    |
    v
+---------------------------+
| 3. glue_schema_validation | -> Column type validation, length profiling, metadata JSON
+---------------------------+
    |
    v
+---------------------------+
| 4. glue_anomaly_detection | -> Length outlier detection, star rating clamping (1 to 5)
+---------------------------+
    |
    v
+---------------------------+
| 5. glue_sampling          | -> Stratified downsampling to 50,000 balanced binary records
+---------------------------+
    |
    v
[Silver Clean Training Data]
```

### 3.3 Model Lifecycle Subsystem (AWS SageMaker & MLflow)
- Base Architecture: Hugging Face `distilbert-base-uncased` (66 Million parameters).
- Task: Sequence classification with a linear head mapping hidden states to 2 logit outputs.
- Compute Infrastructure: Amazon SageMaker training jobs powered by NVIDIA GPU instances (`ml.g4dn.xlarge`).
- Hyperparameters:
  - Batch Size: 32
  - Learning Rate: 2e-5 with AdamW optimizer and linear decay
  - Epochs: 3
  - Weight Decay: 0.01
  - Mixed Precision: FP16 enabled
- Experiment Tracking: MLflow integration records training loss, validation accuracy, precision, recall, and F1 score per epoch.
- Model Promotion: `scripts/s3/app_bucket_s3.py` scans `s3://mlops-sentiment-analysis-data/models/` for the latest timestamped checkpoint, extracts metadata, and copies `model.tar.gz` to `s3://mlops-sentiment-app/models/model.tar.gz`.

### 3.4 Serverless Batch Inference Subsystem (AWS ECS Fargate)
When a user uploads a new dataset for sentiment scoring:
1. The FastAPI backend triggers an ECS Fargate task running `sentiment-pipeline:latest`.
2. Container launches `run_all.py` as entrypoint.
3. Subprocess 1 (`run_clean.py`): Uses local PySpark to validate user columns, clean review strings, and filter invalid rows into `processed.csv`.
4. Subprocess 2 (`run_infer.py`):
   - Downloads `model.tar.gz` from the app bucket (cached locally).
   - Loads DistilBERT tokenizer and weights onto available CPU/GPU.
   - Executes batch tokenization (batch size: 64, max length: 128 tokens).
   - Computes softmax probabilities over class logits: 0 (Negative) vs 1 (Positive).
   - Writes `served.csv` containing original fields, `predicted_label`, and `confidence_score`.
   - Emits terminal completion status to `pipeline_status/status.json`.

### 3.5 Web Application Subsystem (FastAPI & Streamlit)
- Backend API (FastAPI):
  - Enforces JWT Bearer token authentication via AWS Cognito on operational routes.
  - Implements offline simulation mode via `MOCK_AUTH=true` and `MOCK_PIPELINE=true` for local development.
  - Exposes interactive OpenAPI specification at `/docs` and `/redoc`.
- Frontend UI (Streamlit):
  - Multi-page architecture:
    - `streamlit_app.py`: Authentication view capturing username/password and maintaining session state.
    - `pages/upload_and_trigger.py`: File upload drag-and-drop, S3 dispatch, ECS execution triggering, and animated polling monitor.
    - `pages/dashboard.py`: Analytical view rendering sentiment share, review length histograms, and tabular sample views.

---

## 4. End-to-End Data Flow Sequences

### 4.1 Batch Inference Execution Flow

```
User (Streamlit)        FastAPI Backend           AWS ECS Fargate           AWS S3 App Bucket
     |                         |                         |                         |
     |--- 1. Upload CSV ------>|                         |                         |
     |                         |--- 2. Write raw.csv ----------------------------->|
     |                         |                         |                         |
     |--- 3. Trigger Task ---->|                         |                         |
     |    (POST with Token)    |--- 4. ecs:RunTask ----->|                         |
     |                         |    (Overrides: user_id) |                         |
     |<- 5. Task Started ------|                         |                         |
     |                         |                         |--- 6. Read raw.csv ---->|
     |                         |                         |<-- 7. Return raw.csv ---|
     |                         |                         |                         |
     |                         |                         |-- 8. Run Spark Clean ---|
     |                         |                         |-- 9. Run Transformer ---|
     |                         |                         |                         |
     |                         |                         |--- 10. Write served --->|
     |                         |                         |--- 11. Write status --->|
     |                         |                         |    ("COMPLETED")        |
     |--- 12. Poll Status -------------------------------------------------------->|
     |<- 13. Status OK ------------------------------------------------------------|
     |                         |                         |                         |
     |--- 14. Request Dash --->|                         |                         |
     |                         |--- 15. Read served.csv -------------------------->|
     |                         |--- 16. Compute Summary JSON --------------------->|
     |<- 17. Return Summary ---|                         |                         |
     |                         |                         |                         |
     |-- 18. Render Charts ----|                         |                         |
```

---

## 5. Security & Governance

### 5.1 Authentication and Access Control
- Authentication Provider: AWS Cognito User Pools.
- Grant Type: `USER_PASSWORD_AUTH` via `boto3.client('cognito-idp')`.
- API Protection: FastAPI `HTTPBearer` security scheme.
- Verification Logic: Every request to `/trigger_pipeline` and `/generate_dashboard` must present a Bearer token. The backend verifies signature, expiration, and issuer claims.

### 5.2 Network & Compute Security
- VPC Isolation: ECS tasks execute within private subnets with egress via NAT Gateway or S3 VPC Endpoints.
- Least Privilege IAM:
  - ECS Task Execution Role: Limited to reading ECR images and emitting CloudWatch logs.
  - ECS Task Role: Limited to `s3:GetObject` and `s3:PutObject` on `arn:aws:s3:::mlops-sentiment-app/*`.
  - API Instance Role: Limited to `ecs:RunTask` on the specific sentiment analysis task definition.

---

## 6. Architectural Trade-offs & Decisions

| Decision Area | Chosen Approach | Alternative Considered | Rationale |
|---|---|---|---|
| Pipeline Orchestration | AWS ECS Fargate on-demand | AWS Lambda | Batch Spark and PyTorch dependencies exceed Lambda 10GB ephemeral limits and 15-minute runtime ceiling. |
| Status Communication | S3 `status.json` polling | Amazon SQS / Redis Broker | Eliminates dedicated broker costs and idle polling infrastructure for low-to-medium volume batch runs. |
| Model Framework | Hugging Face DistilBERT | Full BERT-Large / RoBERTa | DistilBERT provides 97% of BERT's language understanding while running 60% faster with 40% less memory footprint. |
| Data Processing | PySpark on Glue & ECS | Pure Pandas | Spark enables seamless scaling from small departmental CSVs to multi-gigabyte enterprise datasets without code refactoring. |
| Web Gateway | FastAPI | Flask / Django | Native asynchronous execution, automated OpenAPI generation, and performant Pydantic schema validation. |

