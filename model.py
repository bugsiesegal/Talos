from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from input_module import TextInputModule
from output_module import TextOutputModule
from embedding_module import EmbeddingMerger, EmbeddingSplitter
from cognitive_module import TransformerModule
from config import Config


class IntegratedMemoryModel(nn.Module):
    def __init__(self, config):
        super(IntegratedMemoryModel, self).__init__()
        self.config = config
        self.device = config.device

        self.input_modules = nn.ModuleDict()
        self.output_modules = nn.ModuleDict()
        self.embedding_splitter = EmbeddingSplitter(config)
        self.embedding_merger = EmbeddingMerger(config)
        self.cognitive_module = TransformerModule(config)

        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def add_input_module(self, name, module):
        self.input_modules[name] = module

    def add_output_module(self, name, module):
        self.output_modules[name] = module

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(self, x: Dict[str, Tensor]):
        # Check that all input batches have the same size
        batch_size = None
        for key in x.keys():
            if batch_size is None:
                batch_size = x[key].shape[0]
            else:
                assert batch_size == x[key].shape[0], "All input batches must have the same size"

        # Check that hidden state is initialized and has the correct size
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(batch_size, self.config.max_seq_len,
                                            self.config.embed_dim, device=self.device)
        else:
            assert self.hidden_state.shape[0] == batch_size, "Hidden state has incorrect size"

        # Loop over all inputs and pass them through their respective input modules
        input_embeddings = []
        for key in x.keys():
            input_embeddings.append(self.input_modules[key](x[key]))

        # Loop over all input embeddings and pass them through the embedding merger
        for i in range(len(input_embeddings)):
            self.hidden_state = self.embedding_merger(input_embeddings[i], self.hidden_state)

        # Pass the merged embedding through the cognitive module
        self.hidden_state = self.cognitive_module(self.hidden_state)

        # Pass the hidden state through the embedding splitter
        self.hidden_state, output_embeddings = self.embedding_splitter(self.hidden_state)

        # Loop over all outputs and pass the output embedding through all output modules
        outputs = {}
        for key in self.output_modules.keys():
            outputs[key] = self.output_modules[key](output_embeddings)

        return outputs


