from transformers import AutoTokenizer

from lightning_model import LightningIntegratedMemoryModelText
from config import Config
from lightning_data import HFStreamedTextDatamodule
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, BatchSizeFinder, LearningRateFinder
import wandb
import torch


# run = wandb.init()
# artifact = run.use_artifact('bugsiesegal/integrated-memory-model/model-hm03722l:v0', type='model')
# artifact_dir = artifact.download()

config = Config()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
config.vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters
config.max_seq_len = 512
config.batch_size = 16
config.learning_rate = 1e-4
config.epochs = 10
config.context_length = 8
config.thinking_steps = 5
config.think_twice = False
config.stream = True

# Model Parameters
config.embed_dim = 128
config.num_heads = 8
config.num_layers = 4
config.hidden_dim = 512
config.dropout = 0.1
config.activation = 'gelu'
config.bias = False


model = LightningIntegratedMemoryModelText(config)

state_dict = torch.load("/home/bugsie/PycharmProjects/Talos/artifacts/model-hm03722l:v0" + "/model.ckpt", map_location=torch.device('cpu'))['state_dict']

model.load_state_dict(state_dict)

model.model.to('cpu')

while True:
    text = input("Enter text: ")
    # Encode the text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    # Get last token of the sequence
    last_token = input_ids[:, -1]
    # Truncate the sequence if it is longer than the context length
    input_ids = input_ids[:, -config.context_length:]
    # Generate the next token
    output = model.model({"text": input_ids})
    # Normalize the logits
    output = torch.softmax(output['text'], dim=-1)
    # Get the probability of the last token
    last_token_prob = output[:, last_token]
    # Get the top 10 tokens and their probabilities
    top10_tokens_and_probs = torch.topk(output, 10, dim=-1)
    # Print the top 10 tokens and their probabilities
    for token, prob in zip(top10_tokens_and_probs.indices.tolist()[0], top10_tokens_and_probs.values.tolist()[0]):
        print(f"{tokenizer.decode(token)}: {prob}")

    print(f"{tokenizer.decode(last_token)}: {last_token_prob.tolist()[0][0]}")
