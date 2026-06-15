# Cassandra Roadmap

> The vision for Cassandra as an agentic ML training framework. This document captures planned functionality. For what exists today, see `README.md`.

## Vision

Cassandra aims to be an LLM-backed agentic ML training framework that orchestrates transformer-based model development on Databricks. It will combine HuggingFace Transformers, Optuna/Ray Tune for HPO, TorchDistributor for distributed execution, LangGraph agents, Unity AI Gateway, and Lakebase for a complete agentic training experience.

## Planned Capabilities

- **Agentic Model Training**: LLM-guided hyperparameter optimization and model selection
- **HuggingFace Native**: Direct integration with transformers, datasets, and training APIs
- **Flexible HPO**: Choice between Optuna (lightweight) or Ray Tune (distributed)
- **Distributed Training**: TorchDistributor for multi-GPU/multi-node execution
- **Dual-Memory System**:
  - Short-term: CheckpointSaver for per-conversation workflow state
  - Long-term: DatabricksStore for semantic search across sessions
- **Unity AI Gateway**: Hot-swappable AI models via ChatDatabricks endpoints
- **MLflow Tracing**: Complete observability for LangChain agents and training runs
- **Budget Monitoring**: Token and time tracking with configurable alerts
- **Session Management**: Resume training sessions across multiple runs
- **Unity Catalog Integration**: Seamless data access and model registration

## Target Architecture

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

## Planned CLI

### 1. Basic Model Training with Optuna

```bash
cassandra train \
  --model "distilbert-base-uncased" \
  --dataset "main.default.customer_reviews" \
  --task "classification" \
  --hpo-engine "optuna" \
  --n-trials 20 \
  --output-path /Workspace/models/sentiment_classifier
```

### 2. Distributed Training with Ray Tune

```bash
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

### 3. Interactive Model Development

```bash
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

### 4. Configuration Management

```bash
cassandra config show
cassandra config set ai.endpoint "databricks-claude-sonnet-4-6"
cassandra config set hpo.engine "optuna"  # or "ray"
cassandra config set budget.max_tokens 100000
cassandra config set budget.max_time_minutes 30
```

### 5. Advanced Training with All Options

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

## Planned Configuration Schema

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

## Planned Project Structure

```
src/cassandra/
├── agents/
│   ├── workflow.py          # LangGraph workflow definition
│   ├── nodes.py             # Workflow node implementations
│   └── state.py             # Agent state schema
├── memory/
│   ├── checkpointer.py      # CheckpointSaver wrapper
│   └── store.py             # DatabricksStore wrapper
├── training/
│   ├── hf_trainer.py        # HuggingFace Trainer wrapper
│   ├── optuna_hpo.py        # Optuna hyperparameter optimization
│   ├── ray_tune.py          # Ray Tune HPO integration
│   └── distributor.py       # TorchDistributor wrapper
├── models/
│   ├── hf_models.py         # HuggingFace model utilities
│   └── gateway.py           # Unity AI Gateway client
├── monitoring/
│   ├── budget.py            # Token/time budget tracking
│   └── tracing.py           # MLflow tracing setup
```

## Phases

### Phase 1: Core Training Infrastructure
- [ ] HuggingFace Transformers integration
- [ ] Optuna HPO for single-node optimization
- [ ] Ray Tune HPO for distributed trials
- [ ] TorchDistributor for multi-GPU/multi-node training
- [ ] LangGraph agentic workflow with dual-memory system
- [ ] Unity AI Gateway and MLflow tracing for the agent loop
- [ ] CLI with `train` and `interactive` commands
- [ ] Advanced training strategies (LoRA, QLoRA, mixed precision)
- [ ] Model checkpointing and resumption
- [ ] Gradient accumulation and optimization

### Phase 2: Expanded Model Support
- [ ] Multi-modal models (CLIP, LLaVA, etc.)
- [ ] Encoder-decoder architectures (T5, BART)
- [ ] Generative models (GPT variants, Llama)
- [ ] Efficient fine-tuning (Adapters, IA³, Prompt tuning)
- [ ] Model quantization and compression

### Phase 3: Advanced HPO & Optimization
- [ ] Multi-objective optimization (Pareto frontiers)
- [ ] Population-based training (PBT)
- [ ] Evolutionary strategies
- [ ] Neural architecture search (NAS)
- [ ] Automated data augmentation

### Phase 4: Production & Deployment
- [ ] Model explainability (SHAP, attention visualization)
- [ ] A/B testing framework for model comparison
- [ ] Online learning and continuous training
- [ ] Model serving integration (Databricks Model Serving)
- [ ] Model monitoring and drift detection
- [ ] Cost optimization (spot instances, auto-scaling)
