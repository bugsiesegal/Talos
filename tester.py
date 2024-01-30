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

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = LightningIntegratedMemoryModelText.load_from_checkpoint(
    checkpoint_path="/home/bugsie/PycharmProjects/Talos/artifacts/model-1r43r7n8:v0/model.ckpt",
)

model.model.to('cpu')

model.model.reset_hidden_state()

while True:
    text = input("Enter text: ")

    output_string = ""
    # Encode the text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    # Loop over the input ids and generate the next token and generate the next token till the end of the sequence
    for i in range(100 + len(input_ids)):
        # Get the context window
        context_window = input_ids[:, i:i + model.config.context_length]
        # Generate the next token
        generated_ids = model.model({"text": context_window})['text'].argmax(dim=-1).unsqueeze(0)
        # If i-context_length is equal to input_ids length, then we need to append the next token to
        # the input_ids.
        if i == len(input_ids[0]) - model.config.context_length:
            input_ids = torch.cat([input_ids, generated_ids], dim=1)
            # Append the generated token to the output string
            output_string += tokenizer.decode(generated_ids[0].tolist())
            print(output_string)


