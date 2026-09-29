"""Current attempt receipt checks for model export; no converter or NPU execution."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
from types import ModuleType
import unittest
from unittest import mock

SKILL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage_state_t10', SKILL / 'scripts' / '_stage_state.py')
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


class StageStateIntegrationTests(unittest.TestCase):
    def fixture(self):
        base = SKILL.parent / 'tmp' / 't10-stage-state'
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix='case-', dir=base))

    def test_model_attempt_must_be_new_and_reload_verified(self):
        root = self.fixture()
        old = Path.cwd()
        os.chdir(root)
        try:
            attempt = Path('out/model/attempt-current')
            attempt.mkdir(parents=True)
            train_run = Path('out/train/runs/p5-current')
            train_run.mkdir(parents=True)
            (train_run / 'train_integrity.json').write_text(
                json.dumps({'run_id': 'p5-current'}), encoding='utf-8')
            assets = Path('out/assets/assets.json')
            assets.parent.mkdir(parents=True)
            assets.write_text(json.dumps({'migration_id': 'migration-current'}), encoding='utf-8')
            receipt = {'schema': 'model_artifact.v1', 'attempt_id': attempt.name,
                       'train_run': {'run_id': 'p5-current'},
                       'migration_id': 'migration-current',
                       'model_artifact_id': 'model-current', 'status': 'RELOAD_VERIFIED',
                       'reload_verified': True}
            (attempt / 'model_artifact.json').write_text(json.dumps(receipt), encoding='utf-8')
            item = {'before_model_attempts': [], 'attempt_dir': str(attempt.resolve())}
            with mock.patch.object(state, 'read', return_value={'run_dir': str(train_run.resolve())}):
                self.assertEqual(state._semantic_status('P9', item), 'PASS')
                self.assertIn(str((attempt / 'model_artifact.json').resolve()),
                              state._dynamic_outputs('P9', item))
                bad = {**receipt, 'status': 'EXPORTED_RELOAD_UNVERIFIED', 'reload_verified': False}
                (attempt / 'model_artifact.json').write_text(json.dumps(bad), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'reload not verified'):
                    state._semantic_status('P9', item)
            with self.assertRaisesRegex(ValueError, 'exactly one new'):
                state._current_attempt('P9', {'before_model_attempts': [attempt.name]})
        finally:
            os.chdir(old)

    def test_scene_attempt_binds_current_model_and_raw_evidence(self):
        with mock.patch.dict(os.environ, {'PIPELINE_FLOW': 'reference'}):
            self.assertEqual(state.dependencies('P10'), ['P9'])
        with mock.patch.dict(os.environ, {'PIPELINE_FLOW': 'inference'}):
            self.assertEqual(state.dependencies('P10'), [])
        root = self.fixture()
        old = Path.cwd()
        os.chdir(root)
        try:
            attempt = Path('out/scene/attempt-current')
            attempt.mkdir(parents=True)
            exported = Path('out/model/attempt-source/export_hf')
            exported.mkdir(parents=True)
            artifact = exported.parent / 'model_artifact.json'
            artifact.write_text('{}', encoding='utf-8')
            image = Path('image.png')
            sop = Path('sop.txt')
            image.write_bytes(b'image')
            sop.write_text('SOP', encoding='utf-8')
            evidence = {}
            for key, name, content in (
                    ('generated_ids', 'generated_ids.json', '{}'),
                    ('raw_generation', 'raw_generation.txt', '{"compliant":"无法判断"}'),
                    ('parsed_result', 'parsed_result.json', '{}')):
                path = attempt / name
                path.write_text(content, encoding='utf-8')
                evidence[key] = {'path': str(path.resolve()), 'sha256': state.digest(path)}
            result = {'schema': 'inference_result.v1', 'inference_run_id': attempt.name,
                      'attempt_dir': str(attempt.resolve()), 'status': 'EVIDENCE_INSUFFICIENT',
                      'execution_state': 'NPU_EXECUTED', 'real_inference_verified': True,
                      'business_validity': 'NOT_EVALUATED', **evidence,
                      'model_artifact': {'path': str(artifact.resolve()),
                                         'sha256': state.digest(artifact),
                                         'model_artifact_id': 'model-current',
                                         'export_inventory_sha256': 'inventory-current'},
                      'inputs': {'image': {'path': str(image.resolve()), 'sha256': state.digest(image)},
                                 'sop': {'path': str(sop.resolve()), 'sha256': state.digest(sop)}}}
            (attempt / 'inference_result.json').write_text(json.dumps(result), encoding='utf-8')
            scene_module = ModuleType('_scene_inference')
            scene_module.verify_artifact = lambda *_: (
                {'model_artifact_id': 'model-current'}, {'sha256': 'inventory-current'},
                state.digest(artifact))
            env = {'PIPELINE_FLOW': 'inference',
                   'PIPELINE_MODEL_ARTIFACT': str(artifact.resolve()),
                   'PIPELINE_INFER_MODEL_DIR': str(exported.resolve()),
                   'PIPELINE_PROCESSOR_DIR': str(exported.resolve()),
                   'PIPELINE_EXPORT_MODEL': '0'}
            with mock.patch.dict(os.environ, env), mock.patch.dict(sys.modules, {'_scene_inference': scene_module}):
                self.assertEqual(state._semantic_status('P10', {'attempt_dir': str(attempt.resolve())}), 'PARTIAL')
                self.assertIn(str((attempt / 'raw_generation.txt').resolve()),
                              state._dynamic_outputs('P10', {'attempt_dir': str(attempt.resolve())}))
                with mock.patch.dict(os.environ, {'PIPELINE_FLOW': 'reference'}), \
                        mock.patch.object(state, 'read', return_value={'attempt_dir': str(exported.parent.resolve())}):
                    self.assertEqual(state._semantic_status('P10', {'attempt_dir': str(attempt.resolve())}), 'PARTIAL')
                with mock.patch.dict(os.environ, {'PIPELINE_FLOW': 'reference'}), \
                        mock.patch.object(state, 'read', return_value={'attempt_dir': str((root / 'other-attempt').resolve())}):
                    with self.assertRaisesRegex(ValueError, 'current P9 export'):
                        state._semantic_status('P10', {'attempt_dir': str(attempt.resolve())})
                (attempt / 'raw_generation.txt').write_text('tampered', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'raw/parsed evidence'):
                    state._semantic_status('P10', {'attempt_dir': str(attempt.resolve())})
            with self.assertRaisesRegex(ValueError, 'exactly one new'):
                state._current_attempt('P10', {'before_scene_attempts': [attempt.name]})
        finally:
            os.chdir(old)


if __name__ == '__main__':
    unittest.main()
