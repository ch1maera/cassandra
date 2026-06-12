"""Cassandra AutoML CLI - Production-ready agentic ML training with HuggingFace, Optuna/Ray Tune, and TorchDistributor.

Cassandra is an intelligent agentic ML training framework that orchestrates transformer-based
model development on Databricks, combining HuggingFace Transformers, Optuna/Ray Tune for HPO,
TorchDistributor for distributed execution, LangGraph agents, Unity AI Gateway, and Lakebase
memory for observability and session persistence.

Key Features:
    - Agentic model training with LLM-guided HPO
    - HuggingFace native: transformers, datasets, and training APIs
    - Flexible HPO: Optuna (lightweight) or Ray Tune (distributed)
    - Distributed training: TorchDistributor for multi-GPU/multi-node
    - Dual-memory system (CheckpointSaver + DatabricksStore)
    - Unity AI Gateway integration with hot-swappable models
    - MLflow tracing for complete observability
    - Budget monitoring (tokens and time)
    - Interactive model development via LangGraph agents

Example:
    Basic model training with Optuna::

        from cassandra.config import CassandraConfig
        from cassandra.agents.workflow import create_training_workflow

        config = CassandraConfig.load()
        workflow = create_training_workflow(config)
        result = workflow.run(
            model_name="bert-base-uncased",
            dataset_source="main.default.training_data",
            task_type="classification",
            hpo_engine="optuna",
            n_trials=20,
        )

    CLI usage::

        $ cassandra train \\
            --model "bert-base-uncased" \\
            --dataset "main.default.training_data" \\
            --task "classification" \\
            --hpo-engine "optuna" \\
            --n-trials 20 \\
            --output-path /Workspace/models/classifier

Attributes:
    __version__ (str): The version string for the package
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
