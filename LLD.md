# Low-Level Design (LLD) Document

Project: Amazon Sentiment Analysis Pipeline and MLOps Platform
Maintainer: Ashutosh Yadav
Version: 2.4.0
Status: Approved

---

## 1. Introduction and Scope

This Low-Level Design (LLD) document outlines the detailed architectural specifications, interface definitions, internal data structures, class designs, algorithm implementations, error protocols, and test harnesses for the Amazon Sentiment Analysis Pipeline.

---

## 2. Web Application Layer Specification

### 2.1 API Gateway (`sentiment-app/backend/main.py`)

The FastAPI application acts as the entrypoint for web clients and downstream orchestration.

#### 2.1.1 Pydantic Data Models

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    id_token: str
    token_type: str
    expires_in: int

class TriggerRequest(BaseModel):
    user_id: str
    file_name: str
    bucket_name: Optional[str] = None

class TriggerResponse(BaseModel):
    status: str
    task_arn: Optional[str] = None
    message: str

class DashboardRequest(BaseModel):
    user_id: str
    file_name: str

class DashboardResponse(BaseModel):
    status: str
    total_reviews: int
    positive_reviews: int
    negative_reviews: int
    positive_percentage: float
    negative_percentage: float
    average_length: float
    length_distribution: Dict[str, int]
```

#### 2.1.2 Endpoint Interface Contracts

| Method | Route | Security Dependency | Request Body | Success Response | Error Codes |
|---|---|---|---|---|---|
| GET | `/` | None | None | `{"message": "Sentiment Analysis API is running"}` | None |
| GET | `/health` | None | None | `{"status": "healthy"}` | None |
| POST | `/login` | None | `LoginRequest` | `LoginResponse` | 400, 401, 500 |
| POST | `/trigger_pipeline` | `Depends(verify_token)` | `TriggerRequest` | `TriggerResponse` | 401, 403, 500 |
| POST | `/generate_dashboard` | `Depends(verify_token)` | `DashboardRequest` | `DashboardResponse` | 401, 403, 404, 500 |

### 2.2 Authentication Provider (`sentiment-app/backend/auth.py`)

#### 2.2.1 Function: `authenticate_user(username: str, password: str) -> Dict[str, Any]`
- Purpose: Communicates with AWS Cognito to validate credentials.
- Inputs:
  - `username` (str): Cognito user identifier or email.
  - `password` (str): User password string.
- Mocking Bypass: If `MOCK_AUTH=true` is set in the environment, the function bypasses external network calls and returns deterministic mock tokens for offline verification:
  - `access_token`: `"mock-access-token-xyz"`
  - `id_token`: `"mock-id-token-abc"`
- Cognito API Call:
  ```python
  client = boto3.client("cognito-idp", region_name=AWS_REGION)
  response = client.initiate_auth(
      ClientId=COGNITO_CLIENT_ID,
      AuthFlow="USER_PASSWORD_AUTH",
      AuthParameters={"USERNAME": username, "PASSWORD": password}
  )
  ```
- Exceptions: Catches `NotAuthorizedException` (401), `UserNotFoundException` (401), and general `ClientError` (500).

#### 2.2.2 Function: `verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]`
- Purpose: FastAPI dependency ensuring inbound requests contain a valid Bearer JWT.
- Header Format: `Authorization: Bearer <JWT>`
- Mocking Bypass: If `MOCK_AUTH=true` or token matches `"mock-access-token-xyz"`, returns `{"sub": "mock-user", "username": "developer"}`.
- Live Verification Flow:
  1. Decodes JWT unverified headers to retrieve `kid` (Key ID).
  2. Fetches Cognito public keys from `https://cognito-idp.{region}.amazonaws.com/{userPoolId}/.well-known/jwks.json`.
  3. Validates RS256 signature using matching public key.
  4. Validates token claims: `exp` (expiration), `iss` (issuer), `client_id`.
  5. Raises `HTTPException(status_code=401, detail="Invalid token")` on failure.

### 2.3 Task Runner & Data Utilities (`sentiment-app/backend/utils.py`)

#### 2.3.1 Function: `trigger_ecs_pipeline(user_id: str, file_name: str) -> Dict[str, Any]`
- Purpose: Invokes AWS ECS Fargate task to process the client CSV.
- Mocking Bypass: If `MOCK_PIPELINE=true` or boto3 client unavailable, returns `{"status": "success", "task_arn": "arn:aws:ecs:mock-region:123456789012:task/mock-task-id"}`.
- Live ECS Parameters:
  ```python
  response = ecs_client.run_task(
      cluster=ECS_CLUSTER,
      taskDefinition=ECS_TASK_DEFINITION,
      launchType="FARGATE",
      count=1,
      networkConfiguration={
          "awsvpcConfiguration": {
              "subnets": ECS_SUBNETS,
              "securityGroups": ECS_SECURITY_GROUPS,
              "assignPublicIp": "ENABLED"
          }
      },
      overrides={
          "containerOverrides": [{
              "name": "sentiment-pipeline-container",
              "environment": [
                  {"name": "USER_ID", "value": user_id},
                  {"name": "FILE_NAME", "value": file_name},
                  {"name": "BUCKET_NAME", "value": APP_BUCKET}
              ]
          }]
      }
  )
  ```

#### 2.3.2 Function: `generate_dashboard(user_id: str, file_name: str) -> Dict[str, Any]`
- Purpose: Retrieves inference results from S3, generates analytics, and saves `summary.json`.
- Dynamic Review Column Resolution Algorithm:
  To prevent `KeyError` regressions when different pipeline versions output varied column names, the resolution inspects candidate keys in order of precedence:
  ```python
  candidate_columns = ["review_clean", "review_body", "review", "text"]
  review_col = None
  for col in candidate_columns:
      if col in df.columns:
          review_col = col
          break
  ```
- Metrics Computed:
  - Total records count: `len(df)`
  - Binary sentiment distribution: Sum and percentages where `predicted_label == 1` vs `predicted_label == 0`.
  - Length distribution bins: Categorized into `0-50`, `51-100`, `101-200`, `201-500`, and `500+` characters.
  - Output Persistence: Uploads JSON to `s3://{APP_BUCKET}/dashboard/{user_id}/summary.json`.

---

## 3. Containerized Inference Subsystem (`sentiment-pipeline/`)

### 3.1 Container Architecture (`Dockerfile`)
- Base Image: `bitnami/spark:3.4.1`
- System Dependencies: Python 3.9+, Java 11 JRE, PyTorch (CPU-optimized), Hugging Face `transformers`, `boto3`, `pandas`.
- Workdir: `/app`
- Scripts Packaged: `run_all.py`, `run_clean.py`, `run_infer.py`.
- Execution Entrypoint: `python /app/run_all.py`

### 3.2 Pipeline Coordinator (`run_all.py`)
- Algorithm:
  1. Reads environment variables: `USER_ID`, `FILE_NAME`, `BUCKET_NAME`.
  2. Updates S3 state: Writes `{"status": "CLEANING", "step": 1, "progress": 25}` to `pipeline_status/status.json`.
  3. Executes `python run_clean.py` via `subprocess.run(..., check=True)`.
  4. Updates S3 state: Writes `{"status": "INFERRING", "step": 2, "progress": 60}` to `pipeline_status/status.json`.
  5. Executes `python run_infer.py` via `subprocess.run(..., check=True)`.
  6. Updates S3 state: Writes `{"status": "COMPLETED", "step": 3, "progress": 100}` to `pipeline_status/status.json`.
  7. On Subprocess Failure: Writes `{"status": "FAILED", "error": str(e)}` to S3 and exits with status 1.

### 3.3 Cleansing Engine (`run_clean.py`)
- Spark Staging Collision Fix:
  To avoid directory write collisions on S3, Spark writes to an isolated staging prefix before final single-file promotion:
  ```
  s3://{BUCKET}/tmp/{USER_ID}/processed_temp/part-00000.csv
           |
           v (Move / Flatten)
  s3://{BUCKET}/processed/{USER_ID}/processed.csv
  ```
- Transformation Pipeline:
  - Drop null values on review body and ID columns.
  - Regex text normalization: Strips HTML entities (`&amp;`, `<br />`), converts to lower case, collapses consecutive whitespace.
  - Short review filtering: Flags reviews with fewer than 3 words as invalid or non-substantive.

### 3.4 Inference Engine (`run_infer.py`)
- Batch Processing Algorithm:
  1. Model Acquisition: Checks `/tmp/model/` for cached weights; if absent, downloads `s3://{BUCKET}/models/model.tar.gz` and untars `pytorch_model.bin`, `config.json`, and tokenizer vocab.
  2. PyTorch Initialization:
     ```python
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     tokenizer = AutoTokenizer.from_pretrained(model_dir)
     model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
     model.eval()
     ```
  3. Inference Loop:
     - Feeds review strings in batches of 64 into `tokenizer(..., truncation=True, max_length=128, padding=True, return_tensors="pt")`.
     - Forward pass with `torch.no_grad()` to compute raw logits.
     - Computes softmax probabilities:
       $$\text{confidence} = \max(\sigma(\text{logits}))$$
       $$\text{predicted\_label} = \arg\max(\text{logits})$$
  4. S3 Output: Writes consolidated output dataframe to `s3://{BUCKET}/served/{USER_ID}/served.csv`.

---

## 4. Offline Data Engineering Specification (`scripts/glue/`)

### 4.1 PySpark Execution Jobs

| Script | Input Path | Output Path | Key Transformations |
|---|---|---|---|
| `glue_data_filtering.py` | `raw/reviews.csv` | `splits/train/`, `splits/test/`, `splits/serve/` | Deterministic hash-based 80/10/10 split on `customer_id`. |
| `glue_data_preprocessing.py` | `splits/train/` | `bronze/clean/train/` | Lowcase, whitespace normalization, removal of carriage returns. |
| `glue_schema_validation.py` | `bronze/clean/train/` | `validated/train/` | Type checking: `star_rating` (integer), `review_body` (string). |
| `glue_anomaly_detection.py` | `validated/train/` | `anomaly_filtered/` | Clamps `star_rating` to $[1, 5]$; removes review length outliers. |
| `glue_sampling.py` | `anomaly_filtered/` | `silver/sampled/train_data_sampled.csv` | Binary sentiment: rating $\le 2 \rightarrow 0$, rating $\ge 4 \rightarrow 1$. Stratified sample to 50k balanced rows. |

### 4.2 Stratified Sampling Logic
- Target Class 0 (Negative): Reviews with `star_rating` $\in \{1, 2\}$.
- Neutral Exclusion: Reviews with `star_rating` $= 3$ are excluded to maximize classifier decision boundary margin.
- Target Class 1 (Positive): Reviews with `star_rating` $\in \{4, 5\}$.
- Sample Balancing: PySpark `sampleBy("sentiment", fractions={0: f0, 1: f1})` ensures exactly 25,000 positive and 25,000 negative training records.

---

## 5. SageMaker Model Lifecycle Specification (`scripts/sage_maker/`)

### 5.1 Training Loop (`train.py`)
- Training Parameters:
  - Learning Rate: $2 \times 10^{-5}$
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)
  - Warmup Ratio: 0.1
  - Evaluation Cadence: Every 500 steps
  - Metric Tracking: MLflow client logs train loss, eval loss, eval accuracy, and eval F1 score.
- Model Artifact Packaging:
  ```
  model.tar.gz
  ├── config.json
  ├── pytorch_model.bin
  ├── special_tokens_map.json
  ├── tokenizer_config.json
  └── vocab.txt
  ```

### 5.2 Evaluation Metrics (`evaluate.py`)
- Metrics evaluated against test split:
  - Accuracy: $\frac{TP + TN}{TP + TN + FP + FN} = 93.14\%$
  - Precision: $\frac{TP}{TP + FP} = 92.80\%$
  - Recall: $\frac{TP}{TP + FN} = 93.30\%$
  - F1 Score: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 93.05\%$

### 5.3 Serving Interface (`inference.py`)
Implements the 4 mandatory Amazon SageMaker PyTorch serving handlers:
1. `model_fn(model_dir)`: Deserializes DistilBERT weights from disk.
2. `input_fn(request_body, request_content_type)`: Parses JSON string array of reviews.
3. `predict_fn(input_data, model)`: Generates tensor predictions.
4. `output_fn(prediction, accept)`: Formats scores and predicted labels into JSON response.

---

## 6. S3 File Schemas and Data Formats

### 6.1 Status Schema (`pipeline_status/status.json`)
```json
{
  "user_id": "usr_94821",
  "file_name": "customer_reviews_q3.csv",
  "status": "COMPLETED",
  "step": 3,
  "total_steps": 3,
  "progress_percentage": 100,
  "updated_at": "2026-11-07T12:00:00Z",
  "error": null
}
```

### 6.2 Served CSV Output Schema (`served.csv`)
| Column Name | Data Type | Description |
|---|---|---|
| `review_id` | String | Unique review hash or identifier |
| `review_body` | String | Original review text |
| `review_clean` | String | Cleaned normalized review text |
| `predicted_label` | Integer | Predicted class (0: Negative, 1: Positive) |
| `confidence_score` | Float | Confidence probability in range [0.5, 1.0] |

---

## 7. Automated Test Suite Specification (`tests/`)

The repository includes a 13-test verification suite executing via PyTest:

| Test File | Test Case | Scope & Verification Criteria |
|---|---|---|
| `test_api.py` | `test_root_endpoint` | Asserts GET `/` returns 200 and online status. |
| `test_api.py` | `test_health_endpoint` | Asserts GET `/health` returns `healthy`. |
| `test_api.py` | `test_login_success` | Validates JWT token generation under mock auth. |
| `test_api.py` | `test_login_invalid_credentials` | Asserts 401 when invalid credentials supplied. |
| `test_api.py` | `test_trigger_pipeline_unauthorized` | Asserts 403/401 when Bearer token is omitted. |
| `test_api.py` | `test_trigger_pipeline_authorized` | Validates ECS Fargate launch response with token. |
| `test_api.py` | `test_generate_dashboard_unauthorized` | Asserts 403/401 without Bearer token. |
| `test_glue_logic.py` | `test_text_normalization` | Verifies whitespace stripping and lowercasing. |
| `test_glue_logic.py` | `test_star_rating_clamping` | Validates clamping logic restricts ratings to [1, 5]. |
| `test_glue_logic.py` | `test_binary_sentiment_labeling` | Asserts 1-2 stars map to 0 and 4-5 stars map to 1. |
| `test_glue_logic.py` | `test_short_review_detection` | Validates outlier and short review rejection. |
| `test_inference_logic.py` | `test_label_mapping` | Asserts binary class indices map to Sentiment classes. |
| `test_inference_logic.py` | `test_summary_metrics_calculation` | Asserts sentiment ratios and length bins calculate accurately. |
