import wandb

from run import run

sweep_config = {
    "name": "IntegratedMemoryModelV1",
    "method": "bayes",
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "max_seq_len": {"values": [32,]},
        "context_length": {"values": [4,]},
        "thinking_steps": {"values": [1, 2, 3, 4, 5, 10]},
        "think_twice": {"values": [False, True]},
        "embed_dim": {"values": [128, 256, 512, 1024, 2048]},
        "num_heads": {"values": [8, 16, 32, 64]},
        "num_layers": {"values": [4, 8, 16, 32]},
        "hidden_dim": {"values": [256, 512, 1024, 2048]},
        "max_time_hours": {"values": [1]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="integrated-memory-model")

wandb.agent(sweep_id, function=run, count=10)
