"""Workspace relocation tests; all written fixtures are retained in tmp/."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS=Path(__file__).resolve().parents[1]
ROOT=TOOLS.parent
SCRIPTS=('prepush_check.py','check_report_pdf.py','safe_pack_sync.py','evidence_manifest.py',
         'check_pitfalls.py','negative_controls.py','check_source_consistency.py','align_audit.py',
         'hash_consistency.py','_append_pitfall.py')

class PathTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'tmp'/'path_tests'
        base.mkdir(parents=True,exist_ok=True)
        self.work=Path(tempfile.mkdtemp(prefix='路径 空格 ',dir=base))
        self.env={k:v for k,v in os.environ.items() if not k.startswith('ZERO_RISK_')}
        self.env['PYTHONIOENCODING']='utf-8'

    def run_tool(self,name,*args,env=None):
        return subprocess.run([sys.executable,str(TOOLS/name),*map(str,args)],cwd=self.work,
            env=env or self.env,capture_output=True,text=True,encoding='utf-8',timeout=30)

    def test_all_tools_discover_root_independent_of_cwd(self):
        for name in SCRIPTS:
            with self.subTest(name=name):
                result=self.run_tool(name,'--show-paths')
                self.assertEqual(result.returncode,0,result.stderr)
                paths=json.loads(result.stdout)
                self.assertEqual(Path(paths['root']),ROOT)
                self.assertEqual(Path(paths['skill']),ROOT/'02_活副本_Skill')

    def test_explicit_root_resets_inherited_components(self):
        env={**self.env,'ZERO_RISK_SKILL':str(self.work/'wrong')}
        result=self.run_tool('prepush_check.py','--root',self.work,'--skill','custom skill','--show-paths',env=env)
        paths=json.loads(result.stdout)
        self.assertEqual(Path(paths['skill']),self.work/'custom skill')
        self.assertEqual(Path(paths['evidence']),self.work/'04_远端证据')

    def test_child_gate_uses_explicit_evidence(self):
        ev=self.work/'custom evidence'
        ev.mkdir()
        (ev/'sample.txt').write_text('test evidence',encoding='utf-8')
        result=self.run_tool('evidence_manifest.py','--dir',ev,'--write')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        args=('--root',self.work,'--tools',TOOLS,'--evidence',ev,'--only','G12')
        result=self.run_tool('prepush_check.py',*args)
        self.assertIn('PASS=1 FAIL=0',result.stdout)
        (ev/'sample.txt').write_text('changed evidence',encoding='utf-8')
        result=self.run_tool('prepush_check.py',*args)
        self.assertIn('PASS=0 FAIL=1',result.stdout)

    def test_pitfall_path_does_not_need_source_rewrite(self):
        sample=self.work/'bad.md'
        sample.write_text('| **1** | a |\n| **1** | b |\n',encoding='utf-8')
        result=self.run_tool('check_pitfalls.py','--path',sample)
        self.assertEqual(result.returncode,1)
        self.assertIn('PITFALL_CHECK_FAIL',result.stdout)
        self.assertIn(str(sample),result.stdout)

    def test_pack_apply_blocked_before_writes(self):
        target=self.work/'absent delivery'
        result=self.run_tool('safe_pack_sync.py','--deliverables',target,'--apply')
        self.assertEqual(result.returncode,2)
        self.assertIn('SAFE_PACK_BLOCKED',result.stdout)
        self.assertFalse(target.exists())

    def test_nonexistent_delivery_is_not_replaced_by_default(self):
        result=self.run_tool('prepush_check.py','--deliverables',self.work/'missing','--only','G8')
        self.assertIn('PASS=0 FAIL=1',result.stdout)
        result=self.run_tool('check_report_pdf.py','--deliverables',self.work/'missing')
        self.assertEqual(result.returncode,1)
        self.assertIn('REPORT_PDF_FAIL',result.stdout)

if __name__=='__main__': unittest.main()
