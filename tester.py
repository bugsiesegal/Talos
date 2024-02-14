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
# artifact = run.use_artifact('bugsiesegal/model-registry/IntegratedMemoryModelV1:v0', type='model')
# artifact_dir = artifact.download()

config = Config()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config.vocab_size = tokenizer.vocab_size

model = LightningIntegratedMemoryModelText(config)

# model = LightningIntegratedMemoryModelText.load_from_checkpoint(
#     checkpoint_path="/home/bugsie/PycharmProjects/Talos/artifacts/model-1r43r7n8:v0/model.ckpt",
# )

model.model.to('cpu')

model.model.reset_hidden_state()

while True:
    text = input("Enter text: ")

    output_string = ""
    # Encode the text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    # Generate the output
    generated_ids = model.generate(input_ids, strategy="top_k", top_k=10, temperature=0.7)
    # Decode the output
    output_string += tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Print the output
    print(output_string)


