import importlib.util
import ast
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
MODELS = ('qwen2_audio', 'qwen25_omni', 'llamaomni2')


def load_module(model, output_dir):
    directory = ROOT / 'advwave_defense' / model / 'advwave'
    package = types.ModuleType('advwave')
    package.__path__ = [str(directory)]
    sys.modules['advwave'] = package
    for name in ('advwave.paths', 'advwave.tts', 'advwave.data'):
        sys.modules.pop(name, None)
    paths = types.ModuleType('advwave.paths')
    paths.DATA_DIR = str(DATA)
    paths.OUTPUT_DIR = output_dir
    sys.modules['advwave.paths'] = paths
    tts = types.ModuleType('advwave.tts')
    tts.prompt2audio = lambda prompt, path, **kwargs: Path(path).write_text(prompt, encoding='utf-8')
    sys.modules['advwave.tts'] = tts
    spec = importlib.util.spec_from_file_location('advwave.data', directory / 'data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdvBenchDataTest(unittest.TestCase):
    def test_split_integrity(self):
        def read(name):
            with (DATA / name).open(encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        train = read('advbench_train_320.jsonl')
        test = read('advbench_infer_200.jsonl')
        full = read('AdvBench_Text.jsonl')
        audio = read('AdvBench_Audio.jsonl')
        by_prompt = {row['prompt']: row for row in full}
        self.assertEqual((len(train), len(test), len(by_prompt)), (320, 200, 520))
        self.assertEqual(len({row['prompt'] for row in train + test}), 520)
        self.assertTrue(all(by_prompt[row['prompt']] == row for row in train + test))
        self.assertEqual(len({row['audio'] for row in audio}), 520)
        for row in audio:
            self.assertEqual({k: row[k] for k in by_prompt[row['prompt']]}, by_prompt[row['prompt']])
            self.assertTrue((DATA / 'advbench_audio' / row['audio']).is_file())

    def test_audio_mapping_across_shuffle_seeds_and_custom_tts(self):
        with (DATA / 'AdvBench_Audio.jsonl').open(encoding='utf-8') as f:
            expected = {row['prompt']: row['audio'] for row in map(json.loads, f)}
        with tempfile.TemporaryDirectory() as tmp:
            for model in MODELS:
                module = load_module(model, tmp)
                for seed in (1, 7):
                    rows, _ = module.load_test_dataset(seed=seed)
                    paths = module.ensure_audio_files(rows, generate_tts=False)
                    self.assertEqual(len(paths), 200)
                    for row, path in zip(rows, paths):
                        self.assertEqual(Path(path).name, expected[row['question']])
                a = {'id': 0, 'question': 'first custom prompt'}
                b = {'id': 0, 'question': 'second custom prompt'}
                pa, pb = module.ensure_audio_files([a, b], generate_tts=True)
                self.assertNotEqual(pa, pb)
                self.assertEqual(Path(pa).read_text(encoding='utf-8'), a['question'])
                with self.assertRaises(FileNotFoundError):
                    module.ensure_audio_files([{'id': 1, 'question': 'missing'}], generate_tts=False)

    def test_qwen2_padding_uses_distinct_values(self):
        import numpy as np

        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)
                self.shape = self.value.shape
                self.ndim = self.value.ndim
                self.dtype = self.value.dtype
                self.device = 'cpu'

            def __getitem__(self, item):
                return Tensor(self.value[item])

            def to(self, device=None):
                return self

        fake_torch = types.SimpleNamespace(
            Tensor=Tensor,
            full=lambda shape, value, **kw: Tensor(np.full(shape, value)),
            cat=lambda tensors, dim=0: Tensor(np.concatenate([t.value for t in tensors], axis=dim)),
        )
        source = (ROOT / 'vector_training/qwen2_audio/trl_text_trainer.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
                   for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('_pad_tokens_to_max_length', 'concatenated_inputs')]
        self.assertEqual(len(methods), 2)
        namespace = {'torch': fake_torch}
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                                  ast.ClassDef(name='Trainer', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<trainer-test>', 'exec'), namespace)
        trainer = namespace['Trainer']()
        trainer.padding_value = 42
        trainer.label_pad_token_id = -100
        batch = {
            'chosen_input_ids': Tensor([[1, 2, 3]]),
            'chosen_attention_mask': Tensor([[1, 1, 1]]),
            'chosen_labels': Tensor([[-100, 2, 3]]),
            'rejected_input_ids': Tensor([[1, 4]]),
            'rejected_attention_mask': Tensor([[1, 1]]),
            'rejected_labels': Tensor([[-100, 4]]),
        }
        result = trainer.concatenated_inputs(batch)
        self.assertEqual(result['concatenated_input_ids'].value[1].tolist(), [1, 4, 42])
        self.assertEqual(result['concatenated_attention_mask'].value[1].tolist(), [1, 1, 0])
        self.assertEqual(result['concatenated_labels'].value[1].tolist(), [-100, 4, -100])


if __name__ == '__main__':
    unittest.main()
