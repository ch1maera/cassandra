# Cassandra AutoML CLI

> Production-ready agentic ML training framework with HuggingFace, Optuna/Ray Tune, and TorchDistributor on Databricks.

## Overview

Cassandra is an intelligent LLM-backed ML training framework that orchestrates transformer-based model development on Databricks. It combines HuggingFace Transformers, Optuna/Ray Tune for HPO, TorchDistributor for distributed execution, LangGraph agents, Unity AI Gateway, and Lakebase for a complete agentic training experience.

## Key Features

- **Agentic Model Training**: LLM-guided hyperparameter optimization and model selection
- **HuggingFace Native**: Direct integration with transformers, datasets, and training APIs
- **Flexible HPO**: Choose between Optuna (lightweight) or Ray Tune (distributed)
- **Distributed Training**: TorchDistributor for multi-GPU/multi-node execution
- **Dual-Memory System**:
  - Short-term: CheckpointSaver for per-conversation workflow state
  - Long-term: DatabricksStore for semantic search across sessions
- **Unity AI Gateway**: Hot-swappable AI models via ChatDatabricks endpoints
- **MLflow Tracing**: Complete observability for LangChain agents and training runs
- **Budget Monitoring**: Token and time tracking with configurable alerts
- **Session Management**: Resume training sessions across multiple runs
- **Unity Catalog Integration**: Seamless data access and model registration

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Cassandra Agentic ML Framework                   │
│                                                                        │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐    │
│  │  LangGraph  │──▶│   HuggingFace    │──▶│  Unity Catalog     │    │
│  │   Agent     │   │  Transformers    │   │ (Data & Models)    │    │
│  └─────────────┘   └──────────────────┘   └────────────────────┘    │
│         │                   │                                         │
│         ▼                   ▼                                         │
│  ┌─────────────┐   ┌──────────────────┐                              │
│  │  Unity AI   │   │   Optuna / Ray   │                              │
│  │  Gateway    │   │   Tune (HPO)     │                              │
│  └─────────────┘   └──────────────────┘                              │
│         │                   │                                         │
│         │                   ▼                                         │
│         │          ┌──────────────────┐                              │
│         │          │ TorchDistributor │                              │
│         │          │ (Multi-GPU/Node) │                              │
│         │          └──────────────────┘                              │
│         │                   │                                         │
│         ▼                   ▼                                         │
│  ┌──────────────────────────────────────────┐                        │
│  │          MLflow Tracking & Tracing        │                        │
│  └──────────────────────────────────────────┘                        │
│                     │                                                 │
│                     ▼                                                 │
│  ┌──────────────────────────────────────────┐                        │
│  │            Lakebase Memory                │                        │
│  │  ┌─────────────┐   ┌──────────────────┐  │                        │
│  │  │ Checkpoint  │   │  Databricks      │  │                        │
│  │  │   Saver     │   │    Store         │  │                        │
│  │  └─────────────┘   └──────────────────┘  │                        │
│  └──────────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cassandra.git
cd cassandra

# Install with core dependencies (synced to DBR 18 ML)
pip install -e .

# Install with GPU support (optional)
pip install -e ".[gpu]"

# Install with development tools (optional)
pip install -e ".[dev]"
```

### Prerequisites

1. **Databricks Workspace**: Access to a Databricks workspace with Unity Catalog
2. **Environment Variables**: Configure authentication
   ```bash
   export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
   export DATABRICKS_TOKEN="dapi..."
   export DATABRICKS_WAREHOUSE_ID="your-warehouse-id"
   ```
3. **AI Gateway Endpoint**: Create or access a Unity AI Gateway endpoint (e.g., `databricks-claude-sonnet-4-6`)

### Basic Usage

#### 1. Basic Model Training with Optuna

```bash
# Train a transformer model with Optuna HPO
cassandra train \
  --model "distilbert-base-uncased" \
  --dataset "main.default.customer_reviews" \
  --task "classification" \
  --hpo-engine "optuna" \
  --n-trials 20 \
  --output-path /Workspace/models/sentiment_classifier
```

#### 2. Distributed Training with Ray Tune

```bash
# Train with Ray Tune for parallel HPO across multiple trials
cassandra train \
  --model "bert-base-uncased" \
  --dataset "main.default.text_corpus" \
  --task "sequence-classification" \
  --hpo-engine "ray" \
  --n-trials 50 \
  --distributed \
  --gpus-per-trial 2 \
  --session-id "bert_training_2024"
```

#### 3. Interactive Model Development

```bash
# Start an interactive agentic session
cassandra interactive \
  --dataset "main.default.training_data" \
  --session-id "interactive_dev_2024"
```

Example interaction:
```
> Analyze the dataset distribution
> Suggest appropriate model architectures for this task
> What hyperparameters should we tune?
> Start training with recommended config
> Show training metrics and suggest improvements
```

#### 4. Configuration Management

```bash
# View current configuration
cassandra config show

# Set default AI Gateway endpoint
cassandra config set ai.endpoint "databricks-claude-sonnet-4-6"

# Set HPO engine preference
cassandra config set hpo.engine "optuna"  # or "ray"

# Set budget limits
cassandra config set budget.max_tokens 100000
cassandra config set budget.max_time_minutes 30
```

#### 5. Advanced Training with All Options

```bash
cassandra train \
  --model "roberta-large" \
  --dataset "main.default.training_data" \
  --task "token-classification" \
  --hpo-engine "ray" \
  --n-trials 100 \
  --distributed \
  --num-workers 4 \
  --gpus-per-trial 4 \
  --ai-endpoint "databricks-claude-sonnet-4-6" \
  --lakebase-endpoint "lakebase-prod" \
  --session-id "roberta_ner_2024" \
  --resume \
  --max-tokens 50000 \
  --max-time 120 \
  --output-path /Workspace/models/ner_model \
  --mlflow-experiment "/Shared/cassandra-experiments"
```

## Configuration

### Environment Variables

```bash
# Databricks Configuration
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_WAREHOUSE_ID="your-warehouse-id"

# Unity AI Gateway
export DATABRICKS_AI_GATEWAY_ENDPOINT="databricks-claude-sonnet-4-6"

# Lakebase Configuration
export LAKEBASE_ENDPOINT_URL="https://lakebase-prod.cloud.databricks.com"
export LAKEBASE_CONNECTION_STRING="databricks://lakebase-prod"

# MLflow Configuration
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_NAME="/Shared/cassandra-experiments"

# Budget Limits
export CASSANDRA_MAX_TOKENS=100000
export CASSANDRA_MAX_TIME_MINUTES=30
```

### Configuration File

Create `~/.config/cassandra/config.yaml`:

```yaml
databricks:
  host: "https://your-workspace.cloud.databricks.com"
  warehouse_id: "your-warehouse-id"

ai_gateway:
  endpoint: "databricks-claude-sonnet-4-6"
  temperature: 0.7
  max_tokens: 4096

lakebase:
  endpoint_url: "https://lakebase-prod.cloud.databricks.com"
  connection_string: "databricks://lakebase-prod"
  namespace: "cassandra"
  async: true

mlflow:
  tracking_uri: "databricks"
  experiment_name: "/Shared/cassandra-experiments"
  autolog: true

hpo:
  engine: "optuna"  # or "ray"
  n_trials: 50
  timeout_per_trial: 3600
  direction: "maximize"  # or "minimize"

training:
  batch_size: 32
  learning_rate: 5e-5
  num_epochs: 3
  warmup_steps: 500
  weight_decay: 0.01

distributed:
  use_torch_distributor: true
  num_workers: 4
  gpus_per_worker: 1

budget:
  max_tokens: 100000
  max_time_minutes: 30
  alert_threshold: 0.8
```

## Project Structure

```
cassandra/
├── src/cassandra/
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Click CLI commands
│   ├── config.py                # Configuration management
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── workflow.py          # LangGraph workflow definition
│   │   ├── nodes.py             # Workflow node implementations
│   │   └── state.py             # Agent state schema
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── checkpointer.py      # CheckpointSaver wrapper
│   │   └── store.py             # DatabricksStore wrapper
│   ├── training/
│   │   ├── __init__.py
│   │   ├── hf_trainer.py        # HuggingFace Trainer wrapper
│   │   ├── optuna_hpo.py        # Optuna hyperparameter optimization
│   │   ├── ray_tune.py          # Ray Tune HPO integration
│   │   └── distributor.py       # TorchDistributor wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hf_models.py         # HuggingFace model utilities
│   │   └── gateway.py           # Unity AI Gateway client
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py           # Data loading utilities
│   │   └── preprocessing.py     # HF datasets preprocessing
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── budget.py            # Token/time budget tracking
│   │   └── tracing.py           # MLflow tracing setup
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Structured logging
│       └── exceptions.py        # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_workflow.py
│   │   ├── test_optuna_hpo.py
│   │   └── test_ray_tune.py
│   └── integration/
│       ├── test_training_e2e.py
│       ├── test_distributed.py
│       └── test_interactive.py
├── README.md                    # This file
├── claude.md                    # AI assistant context
├── pyproject.toml               # Project configuration
└── .gitignore
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=cassandra --cov-report=html

# Format code
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/
```

### Code Style

- **Formatting**: Black (100 char line length)
- **Linting**: Ruff (E/W/F/I/B/C4/UP/ARG/SIM rules)
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all modules, classes, and functions

### Testing

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests (requires Databricks)
pytest tests/integration/ -m integration

# Run Databricks-specific tests
pytest -m databricks

# Skip slow tests
pytest -m "not slow"
```

### Markers

- `@pytest.mark.slow`: Long-running tests (>5s)
- `@pytest.mark.integration`: Integration tests requiring external services
- `@pytest.mark.databricks`: Tests requiring Databricks workspace

## Compatibility

- **Python**: 3.11+ (compatible with DBR 18 ML)
- **Databricks Runtime**: DBR 18 ML or higher
- **Key Dependencies** (synced to DBR 18 ML):
  - mlflow-skinny==3.8.1
  - langchain==1.0.3
  - langgraph==1.0.3
  - pandas==2.2.3
  - numpy==2.1.3
  - scikit-learn==1.6.1
  - torch>=2.0.0
  - transformers>=4.35.0
  - optuna>=3.4.0
  - ray[tune]>=2.8.0

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for planned phases and future work.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Support

- **Documentation**: See `claude.md` for detailed architecture and patterns
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join conversations in GitHub Discussions

## Acknowledgments

Built with:
- [HuggingFace Transformers](https://huggingface.co/transformers/) - State-of-the-art NLP models
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework
- [Ray Tune](https://docs.ray.io/en/latest/tune/) - Scalable distributed HPO
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [LangChain](https://python.langchain.com/) - LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent workflows
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Databricks](https://databricks.com/) - Data and AI platform
