"""Synthetic T07 P2/P4/P5 identity pairing; no model or NPU is used."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

import yaml

from test_stage_contracts import load_script, test_directory


binding = load_script('_judge_binding.py')
import _asset_integrity as asset_integrity
from data_id import identity_of_file

BASELINE = 'qwen35-0p8b-triton-20260724-100step-v1'
REFERENCE_SHA = 'reference-log-fixture-sha'
TARGET = '5b5505331924634da64e3d9a1925d02b10babe9f'


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    return path


def rows_digest(rows):
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def inventory(root, names):
    rows = [{'path': name, 'bytes': (root / name).stat().st_size,
             'sha256': binding.digest(root / name)} for name in names]
    return rows, rows_digest(rows)


def fixture(root):
    root = Path(root)
    p2 = root / 'p2'
    run_dir = root / 'runs/p5-synthetic'
    model = root / 'model'
    hf, dcp = model / 'hf', model / 'dcp'
    data_dir = root / 'data'
    for directory in (p2, run_dir, hf, dcp / 'release', data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'model.safetensors.index.json',
                 'preprocessor_config.json', 'tokenizer_config.json',
                 'tokenizer.json', 'model.safetensors'):
        (hf / name).write_bytes(('fixture-' + name).encode())
    (hf / 'tokenizer.json').write_text('{"model":{}}', encoding='utf-8')
    for name in ('latest_checkpointed_iteration.txt', 'release/.metadata',
                 'release/model.distcp'):
        path = dcp / name
        path.write_bytes(b'release' if name.endswith('.txt') else b'fixture payload')
    source_json = write_json(data_dir / 'llava.json',
                             [{'conversations': [{'from': 'human', 'value': 'q'}]}])
    converted = write_json(data_dir / 'converted.json',
                           [{'images': [], 'messages': [{'role': 'user', 'content': 'q'}]}])
    data_id = identity_of_file(str(converted))
    source_id = identity_of_file(str(source_json))
    document = {'training': {'load': str(dcp), 'train_iters': 2, 'save': './save_path'},
                'model': {'model_name_or_path': str(hf)},
                'data': {'dataset_param': {
                    'basic_parameters': {'dataset': str(converted), 'dataset_dir': str(data_dir)},
                    'preprocess_parameters': {'model_name_or_path': str(hf)}}}}
    source = p2 / 'train_config.yaml'
    source.write_text(yaml.safe_dump(document, allow_unicode=True), encoding='utf-8')
    snapshot = run_dir / 'effective_config.yaml'
    document['training']['save'] = str(run_dir / 'checkpoints')
    snapshot.write_text(yaml.safe_dump(document, allow_unicode=True), encoding='utf-8')
    reference = p2 / 'reference_config.yaml'
    reference.write_text('training: {}\n', encoding='utf-8')
    plan = {'schema': 'migrator_config.v1', 'baseline_id': BASELINE,
            'baseline_source': {'sha256': REFERENCE_SHA}, 'target_framework_commit': TARGET,
            'role': 'candidate',
            'effective_config': {'path': str(source), 'sha256': binding.digest(source)},
            'reference_config': {'path': str(reference), 'sha256': binding.digest(reference)}}
    plan_path = write_json(p2 / 'config_manifest.json', plan)
    migration = {'migration_id': 'migration-synthetic', 'target': {'commit': TARGET}}
    migration_path = write_json(root / 'migration_manifest.json', migration)
    hf_rows, hf_hash = inventory(hf, sorted(p.name for p in hf.iterdir()))
    dcp_rows, dcp_hash = inventory(dcp, ['latest_checkpointed_iteration.txt',
                                          'release/.metadata', 'release/model.distcp'])
    receipt = {'schema': 'qwen35_0p8b_conversion.v3', 'migration_id': migration['migration_id'],
               'migration_manifest_sha256': binding.digest(migration_path),
               'hf_file_inventory_sha256': hf_hash, 'dcp_file_inventory_sha256': dcp_hash,
               'target_commit': TARGET}
    receipt_path = write_json(dcp / 'migration_conversion_receipt.json', receipt)
    assets = {'schema': 'migrator_assets.v2', 'attempt_id': 'asset-synthetic',
              'readiness': 'local_complete_official_unverified', 'missing_required': [],
              'migration_identity_state': 'validated_overlay_and_target',
              'migration_id': migration['migration_id'],
              'migration_manifest_path': str(migration_path),
              'migration_manifest_sha256': binding.digest(migration_path),
              'assets': {'weight_hf': {'path': str(hf)}, 'weight_dcp': {'path': str(dcp)},
                         'llava_json': {'path': str(source_json)},
                         'converted_json': {'path': str(converted),
                                            'sha256': data_id['json_sha256'],
                                            'order_sha256': data_id['order_sha256']},
                         'coco_images': {'path': str(data_dir / 'train2017')}},
              'hf_identity': {'status': 'local_complete', 'file_inventory': hf_rows,
                              'file_inventory_sha256': hf_hash},
              'dcp_identity': {'status': 'local_structure_and_hashes', 'file_inventory': dcp_rows,
                               'file_inventory_sha256': dcp_hash},
              'llava_identity': {'status': 'local_content_verified',
                                 'json_sha256': source_id['json_sha256'],
                                 'order_sha256': source_id['order_sha256']},
              'data_conversion': {'source_json_sha256': source_id['json_sha256']},
              'data_identity': {'status': 'local_content_verified',
                                'json_sha256': data_id['json_sha256'],
                                'order_sha256': data_id['order_sha256'],
                                'row_count': data_id['row_count'],
                                'image_inventory': [], 'referenced_image_count': 0,
                                'referenced_images_sha256': rows_digest([]),
                                'image_decode_verified': True},
              'conversion': {'receipt_sha256': binding.digest(receipt_path)},
              'official_data_identity': {'status': 'unverified'}}
    assets_path = write_json(root / 'assets.json', assets)
    log = root / 'train.log'
    log.write_text('SYNTHETIC TEST ONLY\n', encoding='utf-8')
    save_intent = {'source_present': True, 'source_value': './save_path',
                   'enabled': True, 'effective_present': True,
                   'effective_path': str(run_dir / 'checkpoints'), 'actual_format': 'dcp'}
    run = {'run_id': run_dir.name, 'log': str(log), 'config_snapshot': str(snapshot),
           'source_config_sha256': binding.digest(source),
           'effective_config_sha256': binding.digest(snapshot),
           'checkpoint_save': save_intent}
    write_json(run_dir / 'run.json', run)
    train = {'schema': 'train_integrity.v1', 'state': 'COMPLETE', 'problems': [],
             'train_rc': 0, 'failure_markers': [], 'scope': 'FULL_CONFIGURED_TRAINING',
             'run_id': run_dir.name, 'log': str(log), 'log_sha256': binding.digest(log),
             'source_config': str(source), 'source_config_sha256': binding.digest(source),
             'source_config_sha256_after_run': binding.digest(source),
             'config': str(snapshot), 'config_sha256': binding.digest(snapshot),
             'effective_config_sha256': binding.digest(snapshot),
             'config_snapshot_sha256_after_run': binding.digest(snapshot),
             'checkpoint_save': save_intent}
    train_path = write_json(run_dir / 'train_integrity.json', train)
    return {'plan': plan_path, 'assets': assets_path, 'train': train_path,
            'log': log, 'source': source, 'snapshot': snapshot, 'hf': hf, 'dcp': dcp,
            'migration': migration_path, 'converted': converted}


def check(paths, **overrides):
    report = {'shards': ['model.safetensors'], 'model_revision': 'fixture',
              'config_sha256': 'config', 'index_sha256': 'index',
              'header_keys': 488, 'indexed_keys': 488, 'mtp_source_keys': []}
    storage = {'entry': types.SimpleNamespace(relative_path='model.distcp', offset=0,
                                              length=len(b'fixture payload'))}
    with mock.patch.object(asset_integrity, 'metadata_contract', return_value=(report, {}, {})), \
            mock.patch.object(asset_integrity, 'weight_headers_contract', return_value=report), \
            mock.patch.object(asset_integrity, '_read_dcp_storage_data', return_value=storage), \
            mock.patch.object(asset_integrity, '_decode_image', return_value={'status': 'verified'}):
        return binding.verify_binding(
            overrides.get('plan', paths['plan']), overrides.get('assets', paths['assets']),
            overrides.get('train', paths['train']), overrides.get('log', paths['log']),
            BASELINE, REFERENCE_SHA, overrides.get('config', paths['snapshot']))


class BindingTests(unittest.TestCase):
    def test_current_local_bytes_match_but_official_and_runtime_stay_unverified(self):
        with test_directory() as tmp:
            result = check(fixture(tmp))
            self.assertEqual(result['state'], 'LOCAL_BINDING_VERIFIED', result)
            self.assertEqual(result['runtime_asset_binding'], 'UNVERIFIED')
            self.assertEqual(result['official_asset_identity'], 'UNVERIFIED')

    def test_claimed_ready_does_not_hide_payload_or_data_drift(self):
        for target, expected in [('hf', 'hf_content_mismatch'),
                                 ('dcp', 'dcp_content_mismatch'),
                                 ('converted', 'data_content_or_order_mismatch')]:
            with self.subTest(target=target), test_directory() as tmp:
                paths = fixture(tmp)
                file = (paths[target] / ('model.safetensors' if target == 'hf' else
                                         'release/model.distcp')) if target != 'converted' else paths[target]
                file.write_bytes(file.read_bytes() + b'drift')
                self.assertIn(expected, check(paths)['issues'])

    def test_self_consistent_but_omitted_index_shard_is_rejected(self):
        with test_directory() as tmp:
            paths = fixture(tmp)
            assets = json.loads(paths['assets'].read_text(encoding='utf-8'))
            rows = [row for row in assets['hf_identity']['file_inventory']
                    if row['path'] != 'model.safetensors']
            assets['hf_identity']['file_inventory'] = rows
            assets['hf_identity']['file_inventory_sha256'] = rows_digest(rows)
            receipt_path = paths['dcp'] / 'migration_conversion_receipt.json'
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            receipt['hf_file_inventory_sha256'] = rows_digest(rows)
            write_json(receipt_path, receipt)
            assets['conversion']['receipt_sha256'] = binding.digest(receipt_path)
            write_json(paths['assets'], assets)
            self.assertIn('hf_content_mismatch', check(paths)['issues'])

    def test_self_consistent_empty_image_claim_conflicts_with_json_reference(self):
        with test_directory() as tmp:
            paths = fixture(tmp)
            image = paths['converted'].parent / 'train2017/a.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'synthetic-image')
            write_json(paths['converted'], [{'images': ['train2017/a.jpg'],
                                            'messages': [{'role': 'user', 'content': 'q'}]}])
            observed = identity_of_file(str(paths['converted']))
            assets = json.loads(paths['assets'].read_text(encoding='utf-8'))
            assets['data_identity']['json_sha256'] = observed['json_sha256']
            assets['data_identity']['order_sha256'] = observed['order_sha256']
            assets['assets']['converted_json']['sha256'] = observed['json_sha256']
            assets['assets']['converted_json']['order_sha256'] = observed['order_sha256']
            # A forged summary still claims zero referenced images.
            write_json(paths['assets'], assets)
            self.assertIn('data_content_or_order_mismatch', check(paths)['issues'])

    def test_wrong_config_log_run_and_migration_are_rejected(self):
        changes = {
            'source': ('train', 'source_config_sha256', 'wrong', 'p2_p5_source_config_mismatch'),
            'snapshot': ('train', 'config_sha256', 'wrong', 'p5_effective_config_mismatch'),
            'run': ('train', 'run_id', 'other-run', 'train_run_id_mismatch'),
            'log': ('train', 'log_sha256', 'wrong', 'train_log_mismatch'),
            'migration': ('assets', 'migration_id', 'other-migration', 'migration_identity_mismatch'),
        }
        for name, (artifact, key, value, issue) in changes.items():
            with self.subTest(name=name), test_directory() as tmp:
                paths = fixture(tmp)
                doc = json.loads(paths[artifact].read_text(encoding='utf-8'))
                doc[key] = value
                write_json(paths[artifact], doc)
                self.assertIn(issue, check(paths)['issues'])

    def test_effective_config_asset_paths_and_unexpected_delta_are_rejected(self):
        for field, issue in [('dataset_dir', 'train_image_root_mismatch'),
                             ('preprocess_hf', 'train_hf_path_mismatch'),
                             ('model_hf', 'train_hf_path_mismatch'),
                             ('dataset', 'train_data_path_mismatch'),
                             ('dcp', 'train_weight_path_mismatch')]:
            with self.subTest(field=field), test_directory() as tmp:
                paths = fixture(tmp)
                config = yaml.safe_load(paths['snapshot'].read_text(encoding='utf-8'))
                basic = config['data']['dataset_param']['basic_parameters']
                preprocess = config['data']['dataset_param']['preprocess_parameters']
                if field == 'dataset_dir':
                    basic['dataset_dir'] = str(Path(tmp) / 'wrong-coco')
                elif field == 'preprocess_hf':
                    preprocess['model_name_or_path'] = str(Path(tmp) / 'wrong-hf')
                elif field == 'model_hf':
                    config['model']['model_name_or_path'] = str(Path(tmp) / 'wrong-hf')
                elif field == 'dataset':
                    basic['dataset'] = str(Path(tmp) / 'wrong-data.json')
                else:
                    config['training']['load'] = str(Path(tmp) / 'wrong-dcp')
                paths['snapshot'].write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
                snapshot_sha = binding.digest(paths['snapshot'])
                train = json.loads(paths['train'].read_text(encoding='utf-8'))
                train.update(config_sha256=snapshot_sha,
                             effective_config_sha256=snapshot_sha,
                             config_snapshot_sha256_after_run=snapshot_sha)
                write_json(paths['train'], train)
                run_path = paths['train'].parent / 'run.json'
                run = json.loads(run_path.read_text(encoding='utf-8'))
                run['effective_config_sha256'] = snapshot_sha
                write_json(run_path, run)
                result = check(paths)
                self.assertIn(issue, result['issues'])
                self.assertIn('p5_snapshot_source_delta_mismatch', result['issues'])

    def test_bad_nested_json_type_is_structured_rejection(self):
        with test_directory() as tmp:
            paths = fixture(tmp)
            assets = json.loads(paths['assets'].read_text(encoding='utf-8'))
            assets['assets'] = ['not-a-map']
            write_json(paths['assets'], assets)
            result = check(paths)
            self.assertEqual(result['state'], 'REJECTED')
            self.assertIn('hf_content_mismatch', result['issues'])

    def test_missing_receipt_and_partial_inputs_do_not_claim_binding(self):
        with test_directory() as tmp:
            paths = fixture(tmp)
            partial = check(paths, assets=None)
            self.assertEqual(partial['state'], 'REJECTED')
            self.assertIn('binding_inputs_incomplete', partial['issues'])
            (paths['dcp'] / 'migration_conversion_receipt.json').unlink()
            self.assertIn('conversion_receipt_changed', check(paths)['issues'])

    def test_p7_cli_reports_partial_binding_without_changing_execution_code(self):
        with test_directory() as tmp:
            root = Path(tmp)
            paths = fixture(root)
            skill = Path(__file__).resolve().parents[1]
            log = root / 'cli-train.log'
            log.write_text(
                'SYNTHETIC TEST ONLY -- NOT NPU EVIDENCE\n'
                '[Rank 0 | Local Rank 0] iteration 1/1 | consumed samples: 8 | '
                'elapsed time per iteration (ms): 10 | learning rate: 0 | '
                'global batch size: 8 | loss: 2.0 | grad norm: 1.0 |\n', encoding='utf-8')
            output = root / 'judge'
            command = [sys.executable, str(skill / 'scripts/70_judge.py'), '--log', str(log),
                       '--config', str(skill / 'sk04_judge/tests/fixtures/run_sk01_aligned_config.yaml'),
                       '--world-size', '1', '--config-manifest', str(paths['plan']),
                       '--out', str(output)]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       encoding='utf-8', timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summary = json.loads((output / 'judge_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['execution_state'], 'COMPLETED')
            self.assertEqual(summary['source_binding']['state'], 'REJECTED')
            self.assertIn('binding_inputs_incomplete', summary['numeric_open_gates'])
            self.assertEqual(summary['numeric_validity'], 'SOURCE_BINDING_REJECTED')


if __name__ == '__main__':
    unittest.main()
