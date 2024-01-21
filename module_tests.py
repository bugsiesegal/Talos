import unittest
from argparse import Namespace

import torch
from torch import nn

from cognitive_module import TransformerModule
from embedding_module import EmbeddingMerger, EmbeddingSplitter
from input_module import TextInputModule, InputModule
from output_module import TextOutputModule, OutputModule
from model import IntegratedMemoryModel
from config import Config


class TestTextInputModule(unittest.TestCase):

    def setUp(self):
        # Configuration for TextInputModule
        self.config = Config()

    def test_initialization(self):
        # Test initialization
        text_input_module = TextInputModule(self.config)
        self.assertIsInstance(text_input_module, InputModule)

    def test_forward_pass(self):
        # Test forward pass
        text_input_module = TextInputModule(self.config)
        input_tensor = torch.randint(0, self.config.vocab_size, (32, 100))  # Example input
        output = text_input_module(input_tensor)
        self.assertEqual(output.shape, (32, 100, self.config.embed_dim))  # Checking output shape

    def test_output_consistency(self):
        # Test output consistency
        text_input_module = TextInputModule(self.config)
        input_tensor = torch.randint(0, self.config.vocab_size, (1, 10))
        output1 = text_input_module(input_tensor)
        output2 = text_input_module(input_tensor)
        self.assertTrue(torch.equal(output1, output2))

    def test_device_compatibility(self):
        # Test compatibility with different devices
        text_input_module = TextInputModule(self.config).to('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = torch.randint(0, self.config.vocab_size, (1, 10)).to(text_input_module.device)
        output = text_input_module(input_tensor)
        self.assertEqual(output.device, input_tensor.device)

    def test_boundary_conditions(self):
        # Test boundary conditions
        text_input_module = TextInputModule(self.config)
        min_input_tensor = torch.randint(0, self.config.vocab_size, (1, 1))  # Min length
        max_input_tensor = torch.randint(0, self.config.vocab_size, (1, self.config.max_seq_len))  # Max length
        output_min = text_input_module(min_input_tensor)
        output_max = text_input_module(max_input_tensor)
        self.assertEqual(output_min.shape, (1, 1, self.config.embed_dim))
        self.assertEqual(output_max.shape, (1, self.config.max_seq_len, self.config.embed_dim))


class TestTextOutputModule(unittest.TestCase):

    def setUp(self):
        # Configuration for TextOutputModule
        self.config = Config()

    def test_initialization(self):
        # Test initialization
        text_output_module = TextOutputModule(self.config)
        self.assertIsInstance(text_output_module, OutputModule)

    def test_forward_pass(self):
        # Test forward pass
        text_output_module = TextOutputModule(self.config)
        input_tensor = torch.randn(32, 100, self.config.embed_dim)  # Example input
        output = text_output_module(input_tensor)
        self.assertEqual(output.shape, (32, 100, self.config.vocab_size))  # Checking output shape

    def test_output_consistency(self):
        # Test output consistency
        text_output_module = TextOutputModule(self.config)
        input_tensor = torch.randn(1, 10, self.config.embed_dim)
        output1 = text_output_module(input_tensor)
        output2 = text_output_module(input_tensor)
        self.assertTrue(torch.equal(output1, output2))

    def test_device_compatibility(self):
        # Test compatibility with different devices
        text_output_module = TextOutputModule(self.config).to('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = torch.randn(1, 10, self.config.embed_dim).to(text_output_module.device)
        output = text_output_module(input_tensor)
        self.assertEqual(output.device, input_tensor.device)

    def test_boundary_conditions(self):
        # Test boundary conditions
        text_output_module = TextOutputModule(self.config)
        min_input_tensor = torch.randn(1, 1, self.config.embed_dim)  # Minimum input size
        max_input_tensor = torch.randn(1, 100, self.config.embed_dim)  # Example maximum input size

        output_min = text_output_module(min_input_tensor)
        output_max = text_output_module(max_input_tensor)

        self.assertEqual(output_min.shape, (1, 1, self.config.vocab_size))
        self.assertEqual(output_max.shape, (1, 100, self.config.vocab_size))


class TestTransformerModule(unittest.TestCase):

    def setUp(self):
        # Basic configuration for TransformerModule
        self.config = Config()

    def test_initialization(self):
        # Test initialization
        module = TransformerModule(self.config)
        self.assertIsInstance(module, nn.Module)

    def test_forward_pass(self):
        # Test forward pass
        module = TransformerModule(self.config)
        input_tensor = torch.randn(10, 20, self.config.embed_dim)  # Example input
        output = module(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)  # Check if output shape matches input

    def test_run_transformer_encoder(self):
        # Test run_transformer method for encoder
        self.config.transformer_type = "encoder"
        module = TransformerModule(self.config)
        input_tensor = torch.randn(10, 20, self.config.embed_dim)
        mask = torch.zeros(10, 20, dtype=torch.bool)
        output = module.run_transformer(input_tensor, mask, None)
        self.assertEqual(output.shape, input_tensor.shape)

    def test_run_transformer_decoder(self):
        # Test run_transformer method for decoder
        self.config.transformer_type = "decoder"
        module = TransformerModule(self.config)
        input_tensor = torch.randn(10, 20, self.config.embed_dim)
        memory = torch.randn(10, 20, self.config.embed_dim)
        mask = torch.zeros(10, 20, dtype=torch.bool)
        output = module.run_transformer(input_tensor, mask, memory)
        self.assertEqual(output.shape, input_tensor.shape)

    def test_think_twice_decision(self):
        # Test think_twice_decision method
        self.config.think_twice = True
        module = TransformerModule(self.config)
        input_tensor = torch.randn(10, 20, self.config.embed_dim)
        decision = module.think_twice_decision(input_tensor)
        self.assertTrue(decision.dtype == torch.bool)

    def test_device_compatibility(self):
        # Test compatibility with different devices
        module = TransformerModule(self.config).to('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = torch.randn(10, 20, self.config.embed_dim).to(module.device)
        output = module(input_tensor)
        self.assertEqual(output.device, input_tensor.device)

    def test_boundary_conditions(self):
        # Test boundary conditions
        module = TransformerModule(self.config)
        min_input_tensor = torch.randn(1, 1, self.config.embed_dim)
        max_input_tensor = torch.randn(1, 100, self.config.embed_dim)
        output_min = module(min_input_tensor)
        output_max = module(max_input_tensor)
        self.assertEqual(output_min.shape, (1, 1, self.config.embed_dim))
        self.assertEqual(output_max.shape, (1, 100, self.config.embed_dim))


class TestEmbeddingMerger(unittest.TestCase):
    def setUp(self):
        # Basic configuration for EmbeddingMerger
        self.config = Config()

    def test_initialization(self):
        merger = EmbeddingMerger(self.config)
        self.assertIsInstance(merger, nn.Module)

    def test_forward_pass(self):
        merger = EmbeddingMerger(self.config)
        x = torch.randn(10, self.config.embed_dim)
        y = torch.randn(10, self.config.embed_dim)
        output = merger(x, y)
        self.assertEqual(output.shape, (10, self.config.embed_dim))

    def test_device_compatibility(self):
        merger = EmbeddingMerger(self.config).to('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(10, self.config.embed_dim).to(merger.device)
        y = torch.randn(10, self.config.embed_dim).to(merger.device)
        output = merger(x, y)
        self.assertEqual(output.device, x.device)


class TestEmbeddingSplitter(unittest.TestCase):
    def setUp(self):
        # Basic configuration for EmbeddingSplitter
        self.config = Config()

    def test_initialization(self):
        splitter = EmbeddingSplitter(self.config)
        self.assertIsInstance(splitter, nn.Module)

    def test_forward_pass(self):
        splitter = EmbeddingSplitter(self.config)
        x = torch.randn(10, self.config.embed_dim)
        output1, output2 = splitter(x)
        self.assertEqual(output1.shape, (10, self.config.embed_dim))
        self.assertEqual(output2.shape, (10, self.config.embed_dim))

    def test_device_compatibility(self):
        splitter = EmbeddingSplitter(self.config).to('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(10, self.config.embed_dim).to(splitter.device)
        output1, output2 = splitter(x)
        self.assertEqual(output1.device, x.device)
        self.assertEqual(output2.device, x.device)


class TestIntegratedMemoryModel(unittest.TestCase):
    def setUp(self):
        # Basic configuration for IntegratedMemoryModel
        self.config = Config()

    def test_initialization(self):
        model = IntegratedMemoryModel(self.config)
        self.assertIsInstance(model, nn.Module)

    def test_add_input_module(self):
        model = IntegratedMemoryModel(self.config)
        model.add_input_module("text", TextInputModule(self.config))
        self.assertEqual(len(model.input_modules), 1)

    def test_add_output_module(self):
        model = IntegratedMemoryModel(self.config)
        model.add_output_module("text", TextOutputModule(self.config))
        self.assertEqual(len(model.output_modules), 1)

    def test_forward_pass(self):
        model = IntegratedMemoryModel(self.config)
        model.add_input_module("text", TextInputModule(self.config))
        model.add_output_module("text", TextOutputModule(self.config))
        input_tensor = torch.randint(0, self.config.vocab_size, (32, 100))
        output = model({"text": input_tensor})
        self.assertEqual(output.shape, (32, 100, self.config.vocab_size))

    def test_device_compatibility(self):
        model = IntegratedMemoryModel(self.config)
        model.add_input_module("text", TextInputModule(self.config))
        model.add_output_module("text", TextOutputModule(self.config))
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = torch.randint(0, self.config.vocab_size, (32, 100)).to(model.device)
        output = model({"text": input_tensor})
        self.assertEqual(output.device, input_tensor.device)


if __name__ == '__main__':
    unittest.main()
