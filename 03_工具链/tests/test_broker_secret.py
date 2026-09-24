"""Credential configuration tests; no network or credential store writes."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('broker_under_test', ROOT / '05_SSH框架_ops/token_broker.py')
broker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(broker)

class BrokerSecretTests(unittest.TestCase):
    def test_missing_or_blank_configuration_fails_before_side_effects(self):
        for value in (None, '', '   '):
            env = {} if value is None else {'ZERO_RISK_BROKER_SECRET': value}
            with self.subTest(value=value), mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(sys, 'argv', ['broker']), mock.patch.object(broker, 'serve') as serve, \
                    mock.patch.object(broker, 'Broker') as factory, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    broker.main()
                self.assertEqual(error.exception.code, 2)
                serve.assert_not_called()
                factory.assert_not_called()

    def test_environment_and_explicit_override(self):
        for args, expected in (([], 'test-env-only'), (['--secret', 'test-cli-only'], 'test-cli-only')):
            with mock.patch.dict(os.environ, {'ZERO_RISK_BROKER_SECRET': 'test-env-only'}, clear=True), \
                    mock.patch.object(sys, 'argv', ['broker', *args]), mock.patch.object(broker, 'serve') as serve:
                self.assertEqual(broker.main(), 0)
                self.assertEqual(serve.call_args.args[1], expected)

    def test_clients_refuse_missing_secret_without_network(self):
        for name in ('_feedfake.py', '_probehttp.py'):
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch('urllib.request.urlopen') as request:
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(ROOT / '03_工具链' / name), run_name='__main__')
                request.assert_not_called()

if __name__ == '__main__':
    unittest.main()
