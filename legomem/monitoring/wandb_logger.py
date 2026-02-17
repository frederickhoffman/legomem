import os
from typing import Any

import wandb
from dotenv import load_dotenv

load_dotenv()

class WandBLogger:
    def __init__(self, project: str = "legomem", entity: str | None = None):
        self.project = project
        self.entity = entity
        self.run = None

    def start_run(self, config: dict[str, Any], name: str | None = None):
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=config,
            name=name
        )

    def log_metrics(self, metrics: dict[str, Any]):
        if self.run:
            self.run.log(metrics)

    def log_prompt(self, prompt_name: str, prompt_text: str, hyperparameters: dict[str, Any]):
        if self.run:
            # We can log prompts as artifacts or just config
            self.run.config.update({f"prompt_{prompt_name}": prompt_text})
            self.run.config.update({f"hparams_{prompt_name}": hyperparameters})

    def finish_run(self):
        if self.run:
            self.run.finish()
            self.run = None
