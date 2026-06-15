# Cassandra AutoML CLI - AI Assistant Context

> Comprehensive context document for AI assistants working with the Cassandra agentic ML training framework.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Design Decisions](#architecture--design-decisions)
3. [Core Components](#core-components)
4. [Development Guidelines](#development-guidelines)
5. [Integration Points](#integration-points)
6. [Common Patterns](#common-patterns)
7. [Testing Strategy](#testing-strategy)
8. [Deployment & Compatibility](#deployment--compatibility)
9. [Resources](#resources)

---

## Project Overview

### Mission

Cassandra is a production-ready agentic ML training framework that orchestrates transformer-based model development on Databricks, combining HuggingFace Transformers, Optuna/Ray Tune for HPO, TorchDistributor for distributed execution, LangGraph agents, Unity AI Gateway, and Lakebase memory for observability and session persistence.

### Key Capabilities

- **Agentic Model Training**: LLM-guided hyperparameter optimization and model selection
- **HuggingFace Native**: Direct integration with transformers, datasets, and training APIs
- **Flexible HPO**: Choice between Optuna (lightweight, single-node) or Ray Tune (distributed, parallel trials)
- **Distributed Training**: TorchDistributor for multi-GPU/multi-node execution
- **Conversational Development**: Interactive model development via LangGraph agents
- **Dual-Memory System**: Short-term (CheckpointSaver) + long-term (DatabricksStore) persistence
- **Unity AI Gateway**: Hot-swappable AI models with automatic MLflow tracing
- **Budget Monitoring**: Token and time tracking with configurable alerts
- **Session Management**: Resume training workflows across multiple runs

### Target Users

- ML engineers building transformer-based models
- Research scientists exploring new architectures
- Data scientists fine-tuning foundation models
- AI platform teams building training infrastructure

---

## Architecture & Design Decisions

### 1. LangGraph Agent Design

**Pattern**: StateGraph with specialized nodes for agentic training workflow.

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.checkpointer import CheckpointSaver
from cassandra.agents.state import TrainingState
from cassandra.agents.nodes import (
    load_data_node,
    analyze_dataset_node,
    suggest_architecture_node,
    configure_hpo_node,
    run_training_node,
    evaluate_results_node,
    generate_report_node,
)

# Define workflow graph
workflow = StateGraph(TrainingState)

# Add nodes
workflow.add_node("load_data", load_data_node)
workflow.add_node("analyze_dataset", analyze_dataset_node)
workflow.add_node("suggest_architecture", suggest_architecture_node)
workflow.add_node("configure_hpo", configure_hpo_node)
workflow.add_node("run_training", run_training_node)
workflow.add_node("evaluate_results", evaluate_results_node)
workflow.add_node("generate_report", generate_report_node)

# Define edges with conditional routing
workflow.set_entry_point("load_data")
workflow.add_edge("load_data", "analyze_dataset")
workflow.add_conditional_edges(
    "analyze_dataset",
    should_suggest_architecture,
    {
        "suggest": "suggest_architecture",
        "skip": "configure_hpo",
    },
)
workflow.add_edge("suggest_architecture", "configure_hpo")
workflow.add_edge("configure_hpo", "run_training")
workflow.add_edge("run_training", "evaluate_results")
workflow.add_conditional_edges(
    "evaluate_results",
    should_continue_training,
    {
        "continue": "configure_hpo",
        "done": "generate_report",
    },
)

# Compile with checkpointing
checkpointer = CheckpointSaver(connection_string="databricks://lakebase", namespace="cassandra")
agent = workflow.compile(checkpointer=checkpointer)
```

**Why LangGraph?**
- Built-in state management and checkpointing
- Conditional routing for iterative training loops
- Native LangChain/LangSmith integration
- Supports human-in-the-loop interactions

### 2. Dual-Memory System

**Pattern**: CheckpointSaver for short-term state + DatabricksStore for long-term memory.

```python
from langgraph.checkpoint.checkpointer import CheckpointSaver
from langchain_community.storage import DatabricksStore

# Short-term: Per-conversation thread state
checkpointer = CheckpointSaver(
    connection_string="databricks://lakebase",
    namespace="cassandra_checkpoints",
)

# Long-term: Semantic search across all sessions
store = DatabricksStore(
    endpoint_url="https://lakebase-prod.cloud.databricks.com",
    index_name="cassandra_memory",
    text_column="content",
    vector_column="embedding",
)

# Store insights for future retrieval
await store.aset(
    key="forecast_sales_2024_insights",
    value={
        "session_id": "forecast_sales_2024",
        "user_id": "user@example.com",
        "content": "Sales data shows strong weekly seasonality with MASE=0.45",
        "timestamp": "2024-01-15T10:30:00Z",
        "metadata": {"model": "AutoGluon", "dataset": "main.default.sales_data"},
    },
)

# Retrieve relevant memories
results = await store.asimilarity_search(
    query="What were the best performing models for sales forecasting?",
    k=5,
)
```

**Why Dual Memory?**
- **CheckpointSaver**: Fast per-thread state for workflow resumption
- **DatabricksStore**: Semantic search across all user sessions for learning
- Async variants (AsyncCheckpointSaver, AsyncDatabricksStore) for production

### 3. Unity AI Gateway Integration

**Pattern**: Single ChatDatabricks interface with hot-swappable endpoints.

```python
from langchain_community.chat_models import ChatDatabricks

# Initialize with Unity AI Gateway endpoint
llm = ChatDatabricks(
    endpoint="databricks-claude-sonnet-4-6",
    temperature=0.7,
    max_tokens=4096,
)

# Hot-swap to different model
llm_haiku = ChatDatabricks(endpoint="databricks-claude-haiku-4-5")
llm_opus = ChatDatabricks(endpoint="databricks-claude-opus-4-8")

# Use in LangChain chains
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["data_summary"],
    template="Analyze this time series data: {data_summary}",
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(data_summary="...")
```

**Why Unity AI Gateway?**
- Centralized model management
- Built-in rate limiting and cost tracking
- Automatic MLflow tracing via autolog()
- Hot-swappable models without code changes

### 4. HuggingFace + Optuna Integration

**Pattern**: Transformers Trainer with Optuna for lightweight HPO.

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import optuna
import mlflow

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for HPO."""
    # LLM suggests hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 2, 5)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Configure training
    training_args = TrainingArguments(
        output_dir=f"./outputs/trial_{trial.number}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        logging_dir=f"./logs/trial_{trial.number}",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
    )

    # Train with MLflow tracking
    with mlflow.start_run(nested=True):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()

        # Log metrics
        mlflow.log_params(trial.params)
        mlflow.log_metrics(eval_results)

    return eval_results["eval_accuracy"]

# Run Optuna study
with mlflow.start_run():
    study = optuna.create_study(
        direction="maximize",
        study_name="bert_classification_hpo",
    )
    study.optimize(objective, n_trials=20, timeout=3600)

    # Log best trial
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_accuracy", study.best_value)
```

**Why Optuna?**
- Lightweight and fast for single-node HPO
- Excellent for notebook-native workflows
- Pruning algorithms to skip poor trials early
- Easy integration with HuggingFace Trainer

### 5. HuggingFace + Ray Tune Integration

**Pattern**: Transformers with Ray Tune for distributed parallel HPO.

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import mlflow

def train_model(config: dict) -> None:
    """Ray Tune training function."""
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Configure training
    training_args = TrainingArguments(
        output_dir=f"./outputs/{tune.get_trial_id()}",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["num_epochs"],
        warmup_steps=config["warmup_steps"],
        evaluation_strategy="epoch",
        save_strategy="no",  # Ray Tune handles checkpointing
        report_to="none",  # Ray Tune handles reporting
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Report to Ray Tune
    eval_results = trainer.evaluate()
    tune.report(accuracy=eval_results["eval_accuracy"], loss=eval_results["eval_loss"])

# Configure Ray Tune
with mlflow.start_run():
    # LLM suggests search space
    search_space = {
        "model_name": tune.choice(["bert-base-uncased", "roberta-base"]),
        "num_labels": 2,
        "learning_rate": tune.loguniform(1e-5, 5e-5),
        "batch_size": tune.choice([8, 16, 32]),
        "num_epochs": tune.choice([2, 3, 4, 5]),
        "warmup_steps": tune.randint(0, 1000),
    }

    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=5,
        grace_period=1,
        reduction_factor=2,
    )

    # Run Ray Tune
    analysis = tune.run(
        train_model,
        config=search_space,
        num_samples=50,
        scheduler=scheduler,
        resources_per_trial={"cpu": 4, "gpu": 1},
        verbose=1,
    )

    # Log best config
    best_config = analysis.best_config
    mlflow.log_params(best_config)
    mlflow.log_metric("best_accuracy", analysis.best_result["accuracy"])
```

**Why Ray Tune?**
- Distributed parallel trials across multiple nodes/GPUs
- Advanced schedulers (ASHA, HyperBand, PBT)
- Native support for multi-GPU trials
- Scales to hundreds of concurrent trials

### 6. TorchDistributor Integration

**Pattern**: Distributed training for single trial across multiple GPUs/nodes.

```python
from pyspark.ml.torch.distributor import TorchDistributor
from transformers import Trainer, TrainingArguments
import torch.distributed as dist

def train_distributed():
    """Distributed training function for TorchDistributor."""
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()

    # Load model (on each worker)
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=10,
    )

    # Configure for distributed training
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,  # Mixed precision
        ddp_backend="nccl",
        local_rank=local_rank,
    )

    # Train with HF Trainer (handles DDP internally)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Save only on rank 0
    if local_rank == 0:
        trainer.save_model("./final_model")

# Execute with TorchDistributor
distributor = TorchDistributor(
    num_processes=4,  # 4 GPUs
    local_mode=False,  # Multi-node
    use_gpu=True,
)

distributor.run(train_distributed)
```

**Why TorchDistributor?**
- Native Databricks integration
- Handles multi-GPU/multi-node orchestration
- Works seamlessly with Spark clusters
- Simplified distributed training setup

### Decision Matrix: Optuna vs Ray Tune vs TorchDistributor

| Use Case | Recommendation |
|----------|---------------|
| Quick notebook experiments, modest trials | **Optuna** |
| Many parallel HPO trials, each single-GPU | **Ray Tune** |
| Single trial needs multi-GPU/multi-node | **TorchDistributor** |
| Distributed HPO + distributed training | **Ray Tune + TorchDistributor** |
| Team knows PyTorch/HF, wants control | **Optuna + TorchDistributor** |
| Needs advanced schedulers (ASHA, PBT) | **Ray Tune** |

### 7. MLflow Tracing

**Pattern**: autolog() for LangChain + custom spans for HF training.

```python
import mlflow
from mlflow.langchain.autolog import autolog

# Enable automatic tracing for LangChain
autolog()

# Custom spans for HuggingFace training workflow
with mlflow.start_run():
    with mlflow.start_span(name="data_loading") as span:
        dataset = load_dataset_from_unity_catalog(table_name)
        span.set_attribute("num_samples", len(dataset))
        span.set_attribute("num_features", dataset.num_columns)

    with mlflow.start_span(name="tokenization") as span:
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        span.set_attribute("max_length", tokenizer.model_max_length)

    with mlflow.start_span(name="hpo_optimization") as span:
        if hpo_engine == "optuna":
            study = run_optuna_optimization(config)
            span.set_attribute("n_trials", len(study.trials))
            span.set_attribute("best_accuracy", study.best_value)
        elif hpo_engine == "ray":
            analysis = run_ray_tune_optimization(config)
            span.set_attribute("n_trials", len(analysis.trials))
            span.set_attribute("best_accuracy", analysis.best_result["accuracy"])

    with mlflow.start_span(name="final_training") as span:
        trainer = train_final_model(best_config)
        span.set_attribute("model_name", config["model_name"])
        span.set_attribute("training_time_seconds", trainer.state.total_flos)

    with mlflow.start_span(name="evaluation") as span:
        eval_results = trainer.evaluate()
        span.set_attribute("eval_accuracy", eval_results["eval_accuracy"])
        span.set_attribute("eval_loss", eval_results["eval_loss"])
```

**Why MLflow Tracing?**
- Complete observability for LLM and training workflows
- Automatic trace capturing via autolog()
- Custom spans for HuggingFace training steps
- Integration with Databricks workspace
- Parent-child span relationships for nested operations

### 8. Configuration Management

**Pattern**: Hierarchical configuration (code defaults → YAML → env vars → CLI args).

```python
from pydantic import BaseModel, Field
from typing import Literal
import yaml
import os

class HPOConfig(BaseModel):
    engine: Literal["optuna", "ray"] = Field(default="optuna")
    n_trials: int = Field(default=50)
    timeout_per_trial: int = Field(default=3600)
    direction: Literal["maximize", "minimize"] = Field(default="maximize")
    metric: str = Field(default="eval_accuracy")

class TrainingConfig(BaseModel):
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=5e-5)
    num_epochs: int = Field(default=3)
    warmup_steps: int = Field(default=500)
    weight_decay: float = Field(default=0.01)
    fp16: bool = Field(default=True)
    gradient_accumulation_steps: int = Field(default=1)

class DistributedConfig(BaseModel):
    use_torch_distributor: bool = Field(default=False)
    num_workers: int = Field(default=4)
    gpus_per_worker: int = Field(default=1)
    backend: Literal["nccl", "gloo"] = Field(default="nccl")

class CassandraConfig(BaseModel):
    # Databricks
    databricks_host: str = Field(default_factory=lambda: os.getenv("DATABRICKS_HOST"))
    databricks_token: str = Field(default_factory=lambda: os.getenv("DATABRICKS_TOKEN"))
    warehouse_id: str = Field(default_factory=lambda: os.getenv("DATABRICKS_WAREHOUSE_ID"))

    # AI Gateway
    ai_endpoint: str = Field(default="databricks-claude-sonnet-4-6")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)

    # Lakebase
    lakebase_endpoint_url: str = Field(default="https://lakebase-prod.cloud.databricks.com")
    lakebase_namespace: str = Field(default="cassandra")

    # HPO Configuration
    hpo: HPOConfig = Field(default_factory=HPOConfig)

    # Training Configuration
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Distributed Configuration
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    # Budget
    max_tokens: int = Field(default=100000)
    max_time_minutes: int = Field(default=30)

    @classmethod
    def from_yaml(cls, path: str) -> "CassandraConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def load(cls) -> "CassandraConfig":
        # Priority: CLI args > env vars > YAML > defaults
        config_path = os.path.expanduser("~/.config/cassandra/config.yaml")
        if os.path.exists(config_path):
            return cls.from_yaml(config_path)
        return cls()
```

**Why Hierarchical Config?**
- Flexible for different deployment environments
- Pydantic validation ensures correctness
- Easy to override for testing
- Supports both local dev and Databricks notebooks
- Nested configuration for complex training workflows

### 9. Budget Monitoring

**Pattern**: Token and time tracking with configurable alerts.

```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class BudgetTracker:
    max_tokens: int
    max_time_minutes: int
    alert_threshold: float = 0.8

    tokens_used: int = 0
    start_time: datetime = None

    def start(self) -> None:
        self.start_time = datetime.now()

    def track_tokens(self, tokens: int) -> None:
        self.tokens_used += tokens
        if self.tokens_used >= self.max_tokens * self.alert_threshold:
            self._alert("Token budget approaching limit")
        if self.tokens_used >= self.max_tokens:
            raise BudgetExceededError("Token budget exceeded")

    def check_time(self) -> None:
        elapsed = datetime.now() - self.start_time
        max_time = timedelta(minutes=self.max_time_minutes)
        if elapsed >= max_time * self.alert_threshold:
            self._alert("Time budget approaching limit")
        if elapsed >= max_time:
            raise BudgetExceededError("Time budget exceeded")

    def _alert(self, message: str) -> None:
        print(f"[BUDGET ALERT] {message}")
```

---

## Core Components

### CLI Commands (`src/cassandra/cli.py`)

```python
import click
from rich.console import Console

@click.group()
def main():
    """Cassandra Agentic ML Training CLI"""
    pass

@main.command()
@click.option("--model", required=True, help="HuggingFace model name or path")
@click.option("--dataset", required=True, help="Unity Catalog table name")
@click.option("--task", required=True, type=click.Choice(["classification", "regression", "token-classification", "qa"]))
@click.option("--hpo-engine", default="optuna", type=click.Choice(["optuna", "ray"]))
@click.option("--n-trials", type=int, default=50)
@click.option("--distributed/--no-distributed", default=False)
@click.option("--num-workers", type=int, default=4)
@click.option("--gpus-per-worker", type=int, default=1)
@click.option("--session-id", help="Session ID for resumption")
@click.option("--output-path", required=True, help="Output path for model")
def train(
    model: str,
    dataset: str,
    task: str,
    hpo_engine: str,
    n_trials: int,
    distributed: bool,
    num_workers: int,
    gpus_per_worker: int,
    session_id: str,
    output_path: str,
):
    """Train a model with hyperparameter optimization"""
    console = Console()
    console.print(f"[bold]Starting training: {model} on {dataset}[/bold]")
    console.print(f"HPO Engine: {hpo_engine} | Trials: {n_trials}")
    if distributed:
        console.print(f"Distributed: {num_workers} workers × {gpus_per_worker} GPUs")
    # Implementation...

@main.command()
@click.option("--dataset", required=True)
@click.option("--session-id", required=True)
def interactive(dataset: str, session_id: str):
    """Start interactive model development session"""
    console = Console()
    console.print(f"[bold]Starting interactive session for {dataset}[/bold]")
    # Implementation...

@main.command()
@click.argument("action", type=click.Choice(["show", "set", "reset"]))
@click.argument("key", required=False)
@click.argument("value", required=False)
def config(action: str, key: str, value: str):
    """Manage configuration"""
    # Implementation...

@main.command()
@click.option("--session-id", required=True)
def sessions(session_id: str):
    """List and manage training sessions"""
    # Implementation...
```

### Agent State (`src/cassandra/agents/state.py`)

```python
from typing import TypedDict, Optional, List, Dict, Any, Literal
from datetime import datetime

class TrainingState(TypedDict):
    # User inputs
    model_name: str
    dataset_source: str
    task_type: Literal["classification", "regression", "token-classification", "question-answering"]
    hpo_engine: Literal["optuna", "ray"]

    # Workflow state
    current_step: str
    raw_dataset: Optional[Any]
    dataset_analysis: Optional[Dict[str, Any]]
    tokenized_dataset: Optional[Any]
    suggested_architecture: Optional[str]
    hpo_config: Optional[Dict[str, Any]]
    hpo_results: Optional[Dict[str, Any]]
    best_config: Optional[Dict[str, Any]]
    trained_model: Optional[str]
    evaluation_metrics: Optional[Dict[str, float]]
    final_report: Optional[str]

    # Training metadata
    n_trials: int
    n_trials_completed: int
    best_metric: Optional[float]
    training_time_seconds: Optional[float]

    # Metadata
    session_id: str
    user_id: str
    start_time: datetime
    last_updated: datetime

    # Memory
    conversation_history: List[Dict[str, str]]
    retrieved_insights: List[Dict[str, Any]]

    # Distributed training
    use_distributor: bool
    num_workers: int
    gpus_per_worker: int
```

---

## Development Guidelines

### Code Style

- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for modules, classes, functions
- **Formatting**: Black (100 char line length)
- **Linting**: Ruff (E/W/F/I/B/C4/UP/ARG/SIM rules)

Example:
```python
def preprocess_time_series(
    data: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    frequency: str = "D",
) -> TimeSeriesDataFrame:
    """Preprocess raw data into AutoGluon TimeSeriesDataFrame.

    Args:
        data: Raw input DataFrame
        target_column: Name of the target column
        timestamp_column: Name of the timestamp column
        frequency: Frequency of the time series (D, W, M, etc.)

    Returns:
        Preprocessed TimeSeriesDataFrame ready for training

    Raises:
        ValueError: If required columns are missing
    """
    # Implementation...
```

### Error Handling

Define custom exceptions:
```python
# src/cassandra/utils/exceptions.py
class CassandraError(Exception):
    """Base exception for Cassandra"""

class ConfigurationError(CassandraError):
    """Raised when configuration is invalid"""

class BudgetExceededError(CassandraError):
    """Raised when budget limits are exceeded"""

class DataLoadError(CassandraError):
    """Raised when data loading fails"""
```

Use retry logic for external services:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def load_table_from_unity_catalog(table_name: str) -> pd.DataFrame:
    """Load table from Unity Catalog with retry logic"""
    # Implementation...
```

### Logging

Use structured logging:
```python
import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("cassandra")

# Use context in logs
logger.info("Loading data", extra={"table": table_name, "session_id": session_id})
```

---

## Integration Points

### Unity AI Gateway

- **Endpoint**: `databricks-claude-sonnet-4-6` (or custom)
- **Authentication**: Via `DATABRICKS_TOKEN`
- **Rate Limits**: Handled by gateway
- **Tracing**: Automatic via `mlflow.langchain.autolog()`

### Lakebase

- **CheckpointSaver**: `databricks://lakebase` connection string
- **DatabricksStore**: Vector search index in Unity Catalog
- **Namespace**: `cassandra` (or configurable)
- **Async**: Recommended for production (`AsyncCheckpointSaver`, `AsyncDatabricksStore`)

### MLflow

- **Tracking URI**: `databricks` (points to workspace)
- **Experiment**: `/Shared/cassandra-experiments` (default)
- **Autolog**: Enabled for LangChain via `autolog()`
- **Custom Spans**: For AutoGluon and preprocessing steps

### Unity Catalog

- **Data Access**: Via Databricks SDK or Spark SQL
- **Model Storage**: Registered models in Unity Catalog
- **Artifact Logging**: All outputs stored in workspace

---

## Common Patterns

### CLI Command Structure

```python
@main.command()
@click.option("--data", required=True)
@click.option("--session-id", required=True)
@click.pass_context
def my_command(ctx: click.Context, data: str, session_id: str):
    """Command description"""
    config = CassandraConfig.load()
    # Implementation...
```

### Authentication Flow

```python
from databricks.sdk import WorkspaceClient

def get_databricks_client() -> WorkspaceClient:
    """Get authenticated Databricks client"""
    return WorkspaceClient(
        host=config.databricks_host,
        token=config.databricks_token,
    )
```

### Data Loading

```python
from datasets import Dataset, DatasetDict
import pandas as pd

def load_dataset_from_unity_catalog(table_name: str) -> Dataset:
    """Load dataset from Unity Catalog table as HuggingFace Dataset"""
    client = get_databricks_client()

    # Load data via Databricks SQL
    with sql.connect(
        server_hostname=config.databricks_host,
        http_path=f"/sql/1.0/warehouses/{config.warehouse_id}",
        access_token=config.databricks_token,
    ) as connection:
        df = pd.read_sql(f"SELECT * FROM {table_name}", connection)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    return dataset

def tokenize_dataset(dataset: Dataset, tokenizer, text_column: str = "text") -> Dataset:
    """Tokenize dataset for training"""
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset

def create_train_eval_split(dataset: Dataset, test_size: float = 0.2) -> DatasetDict:
    """Split dataset into train and eval"""
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
    return DatasetDict({
        "train": split_dataset["train"],
        "eval": split_dataset["test"],
    })
```

---

## Testing Strategy

### Unit Tests

Mock external dependencies:
```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_llm():
    return Mock(spec=ChatDatabricks)

@pytest.fixture
def mock_config():
    return CassandraConfig(
        databricks_host="https://test.databricks.com",
        databricks_token="test-token",
        warehouse_id="test-warehouse",
    )

def test_load_data_node(mock_config):
    """Test data loading node"""
    # Implementation...
```

### Integration Tests

Mark with `@pytest.mark.integration`:
```python
@pytest.mark.integration
@pytest.mark.databricks
def test_forecast_e2e():
    """End-to-end test requiring Databricks workspace"""
    # Requires real credentials
    # Implementation...
```

### Test Markers

- `@pytest.mark.unit`: Fast unit tests (default)
- `@pytest.mark.slow`: Tests taking >5s
- `@pytest.mark.integration`: Tests requiring external services
- `@pytest.mark.databricks`: Tests requiring Databricks workspace

---

## Deployment & Compatibility

### DBR 18 ML Compatibility

**Critical Dependencies** (exact versions):
- `mlflow-skinny==3.8.1`
- `langchain==1.0.3`
- `langgraph==1.0.3`
- `pandas==2.2.3`
- `numpy==2.1.3`
- `scikit-learn==1.6.1`

**Flexible Dependencies**:
- `autogluon.timeseries>=1.1.0` (not in DBR 18 ML)

**Testing Compatibility**:
1. Create DBR 18 ML cluster
2. Install Cassandra: `pip install -e .`
3. Run integration tests: `pytest -m databricks`

### Environment Setup

```bash
# Local development
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_WAREHOUSE_ID="your-warehouse-id"

# Databricks notebook (authentication automatic)
%pip install cassandra-automl
```

---

## Resources

### Documentation

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/index)
- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/)
- [Ray Tune Guide](https://docs.ray.io/en/latest/tune/index.html)
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [TorchDistributor on Databricks](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/torch-distributor.html)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Databricks](https://python.langchain.com/docs/integrations/chat/databricks)
- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html)
- [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
- [Lakebase Documentation](https://docs.databricks.com/en/lakehouse-ai/lakebase/index.html)

### Related Projects

- [HuggingFace Examples](https://github.com/huggingface/transformers/tree/main/examples)
- [Optuna Examples](https://github.com/optuna/optuna-examples)
- [Ray Tune Examples](https://github.com/ray-project/ray/tree/master/python/ray/tune/examples)
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [Databricks SDK Python](https://github.com/databricks/databricks-sdk-py)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)

### Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Slack**: Join #cassandra-support (if internal)

---

## Future Extensibility

### Phase 2: Expanded Model Support

Add multi-modal and generative models:
```python
# src/cassandra/training/multimodal.py
from transformers import CLIPModel, CLIPProcessor

class MultiModalTrainer(BaseTrainer):
    def train(self, dataset: Dataset) -> CLIPModel:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Training logic...
        return model

# src/cassandra/training/generative.py
from transformers import AutoModelForCausalLM

class GenerativeTrainer(BaseTrainer):
    def train(self, dataset: Dataset) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        # LoRA/QLoRA fine-tuning...
        return model
```

### Phase 3: Advanced HPO Strategies

Add population-based training and NAS:
```python
# src/cassandra/training/pbt.py
from ray.tune.schedulers import PopulationBasedTraining

def run_pbt_optimization(config: dict) -> Analysis:
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_accuracy",
        mode="max",
        perturbation_interval=5,
        hyperparam_mutations={
            "learning_rate": lambda: tune.loguniform(1e-5, 1e-4).sample(),
            "batch_size": lambda: tune.choice([16, 32, 64]).sample(),
        },
    )
    return tune.run(train_function, scheduler=scheduler, num_samples=20)
```

### Phase 4: Production Deployment

Add model serving and monitoring:
```python
# src/cassandra/deployment/serving.py
def deploy_to_model_serving(
    model_path: str,
    endpoint_name: str,
    workload_size: str = "Small",
) -> str:
    """Deploy model to Databricks Model Serving"""
    client = get_databricks_client()
    endpoint = client.serving_endpoints.create(
        name=endpoint_name,
        config={
            "served_models": [{
                "model_name": model_path,
                "model_version": "1",
                "workload_size": workload_size,
                "scale_to_zero_enabled": True,
            }]
        },
    )
    return endpoint.endpoint_url
```
