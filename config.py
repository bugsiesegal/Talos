from dataclasses import dataclass
from transformers import AutoTokenizer
import wandb


@dataclass
class Config:
    def __init__(self):
        # Tokenizer
        self.tokenizer_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer_special_tokens = {
            "pad_token": self.tokenizer.eos_token,
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "unk_token": self.tokenizer.unk_token,
            "mask_token": self.tokenizer.mask_token,
        }
        self.tokenizer.eos_token = self.tokenizer_special_tokens["eos_token"]
        self.tokenizer.bos_token = self.tokenizer_special_tokens["bos_token"]
        self.tokenizer.unk_token = self.tokenizer_special_tokens["unk_token"]
        self.tokenizer.pad_token = self.tokenizer_special_tokens["pad_token"]
        self.tokenizer.mask_token = self.tokenizer_special_tokens["mask_token"]

        # Data
        self.max_length = 1024
        self.batch_size = 16
        self.num_workers = 4
        self.streaming = False
        self.dataset_name = "wikitext"
        self.dataset_subname = "wikitext-103-v1"

        # Training
        self.context_window = 16
        self.learning_rate = 1e-3
        self.lr_scheduler_factor = 0.1
        self.lr_scheduler_patience = 400
        self.min_lr = 1e-6
        self.precision = '16-mixed'
        self.benchmark = True
        self.gradient_clip_val = 0.5
        self.max_time = {'hours': 20, 'minutes': 0}
        self.log_every_n_steps = 10

        # Model
        self.thinking_steps = 8
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding_dim = 512
        self.nhead = 64
        self.dim_feedforward = 2048
        self.preprocessor_layers = 4
        self.core_layers = 4
        self.postprocessor_layers = 4
        self.dropout = 0.1
