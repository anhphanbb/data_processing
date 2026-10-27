"""Storage/orchestration tests using tiny stand-ins, not scientific validation."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run_awe_pipeline_orbit.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.permanent = r.month_paths(self.root/'soc',2023,11)
        self.work = self.root/'work'
        self.logs = self.root/'logs'
        self.logs.mkdir()
        self.group = {}
        self.orbit = '00142'
        for filt in r.FILTERS:
            path = self.permanent['l0c']/f'awe_l0c_{filt}_2023335T0329_00142_v02.nc'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'original')
            self.group[filt] = path
        self.args = argparse.Namespace(start_step=1,save_l0d=False,save_l1f=False,save_l1r=False,workers=4,overwrite=False)
        self.receipt = self.root/'receipt.json'
        self.steps = []

    def fake_stage(self,command,cwd,logfile):
        step = int(logfile.stem.split('_')[1])
        self.steps.append(step)
        if step == 4: self.assertFalse((self.work/'l1f').exists() if not self.args.save_l1f else False)
        if step == 6 and not self.args.save_l1r: self.assertFalse(list((self.work/'l1r').glob('*.nc')))
        levels = {1:['l0d'],2:['l1f'],3:['l1r'],4:['l1c'],5:['l1a'],6:['l2a','l3a','l3c']}[step]
        for level in levels:
            filters = r.FILTERS if level in ('l1f','l1r','l1a') else ('tmp',) if level in ('l2a','l3a') else ('q20',)
            dirs = [self.work/level/f'{x}x{x}' for x in r.RESOLUTIONS] if level in ('l2a','l3a') else [self.work/level]
            for folder in dirs:
                folder.mkdir(parents=True,exist_ok=True)
                for f in filters: (folder/f'awe_{level}_{f}_2023335T0329_00142_v02.nc').write_bytes(b'product')
        if step == 3: (self.work/'l1r'/'Master00142.mgf').write_bytes(struct.pack('<ii',1,1)+bytes(16))

    def fake_validate(self,step,paths,orbit,logdir):
        r.expected_outputs(step,paths,orbit)

    def run_one(self,fail_stage=None,invalid_stage=None):
        def run(*args):
            if fail_stage and args[2].stem == f'step_{fail_stage}': raise RuntimeError('injected processing failure')
            self.fake_stage(*args)
        def validate(*args):
            if args[0] == invalid_stage: raise ValueError('injected validation failure')
            self.fake_validate(*args)
        with patch.object(r,'run_logged',side_effect=run),patch.object(r,'validate_outputs',side_effect=validate):
            r.process_one(self.orbit,self.group,self.args,self.permanent,self.work,self.logs,self.receipt,'signature')

    def test_success_cleanup_master_and_receipt(self):
        self.run_one()
        self.assertEqual(self.steps,[1,2,3,4,5,6])
        self.assertFalse(self.work.exists())
        self.assertFalse(self.permanent['l1f'].exists())
        self.assertEqual([p.name for p in self.permanent['l1r'].iterdir()],['Master00142.mgf'])
        self.assertTrue(r.receipt_matches(self.receipt,'signature'))
        product = next(self.permanent['l3c'].glob('*.nc'))
        product.write_bytes(b'changed')
        self.assertFalse(r.receipt_matches(self.receipt,'signature'))
        self.assertTrue(all(p.read_bytes()==b'original' for p in self.group.values()))

    def test_keep_each_combination(self):
        # Each combination has its own complete workspace and destination tree.
        for keep_f,keep_r in ((True,False),(False,True),(True,True)):
            case = LifecycleTests('test_success_cleanup_master_and_receipt')
            case.setUp()
            try:
                case.args.save_l1f,case.args.save_l1r = keep_f,keep_r
                case.run_one()
                self.assertEqual(len(list(case.permanent['l1f'].glob('*.nc'))),4 if keep_f else 0)
                self.assertEqual(len(list(case.permanent['l1r'].glob('*.nc'))),4 if keep_r else 0)
            finally: case.doCleanups()

    def test_failure_retains_inputs_and_stops(self):
        with self.assertRaises(RuntimeError): self.run_one(fail_stage=3)
        self.assertTrue((self.work/'l1f').exists())
        self.assertFalse((self.work/'l0d').exists())
        self.assertFalse(self.receipt.exists())
        self.assertEqual(self.steps,[1,2])
        self.assertFalse(self.permanent['l3c'].exists())

    def test_failed_validation_does_not_cleanup(self):
        with self.assertRaises(ValueError): self.run_one(invalid_stage=3)
        self.assertEqual(len(list((self.work/'l1f').glob('*.nc'))),4)
        self.assertFalse(self.receipt.exists())

    def test_start_two_preserves_existing_l0d(self):
        self.args.start_step = 2
        source = self.permanent['l0d']/'awe_l0d_q20_2023335T0329_00142_v02.nc'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'original L0D')
        self.group['l0d'] = source
        self.run_one()
        self.assertEqual(self.steps,[2,3,4,5,6])
        self.assertEqual(source.read_bytes(),b'original L0D')

    def test_duplicate_and_missing_filters_rejected(self):
        groups = r.discover_orbits(self.permanent['l0c'])
        self.assertEqual(list(groups),['00142'])
        duplicate = self.group['q20'].with_name(self.group['q20'].name.replace('v02','v03'))
        duplicate.write_bytes(b'other')
        with self.assertRaises(ValueError): r.discover_orbits(self.permanent['l0c'])
        duplicate.unlink()
        self.group['p12'].unlink()
        with self.assertRaises(ValueError): r.discover_orbits(self.permanent['l0c'])

    def test_missing_final_output_prevents_publish(self):
        original = self.fake_validate
        def check(step,*args):
            if step == 6: next((self.work/'l3c').glob('*.nc')).unlink()
            original(step,*args)
        with patch.object(r,'run_logged',side_effect=self.fake_stage),patch.object(r,'validate_outputs',side_effect=check):
            with self.assertRaises(ValueError):
                r.process_one(self.orbit,self.group,self.args,self.permanent,self.work,self.logs,self.receipt,'signature')
        self.assertFalse(self.receipt.exists())
        self.assertTrue((self.work/'l1r'/'Master00142.mgf').exists())

    def test_publish_failure_keeps_workspace(self):
        with patch.object(r,'publish_file',side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.run_one()
        self.assertTrue(self.work.exists())
        self.assertFalse(self.receipt.exists())

    def test_original_mlspb_filename_completes_pipeline(self):
        self.args.save_l0d = True
        original = self.fake_stage
        def stage(command,cwd,logfile):
            original(command,cwd,logfile)
            if logfile.stem == 'step_1':
                f = next((self.work/'l0d').glob('*.nc'))
                f.rename(f.with_name(f.name.replace('_l0d_','_l0c_')))
        with patch.object(self,'fake_stage',side_effect=stage): self.run_one()
        self.assertTrue(r.receipt_matches(self.receipt,'signature'))
        self.assertEqual(len(list(self.permanent['l0d'].glob('awe_l0c_*.nc'))),1)

    def test_preserved_mlcloud_filename_normalized_for_coadd(self):
        original = self.fake_stage
        def stage(command,cwd,logfile):
            original(command,cwd,logfile)
            if logfile.stem == 'step_4':
                f = next((self.work/'l1c').glob('*.nc'))
                f.rename(f.with_name(f.name.replace('_l1c_','_l1r_')))
            if logfile.stem == 'step_5':
                self.assertEqual(len(list((self.work/'l1c').glob('awe_l1c_*.nc'))),1)
                self.assertFalse(list((self.work/'l1c').glob('awe_l1r_*.nc')))
        with patch.object(self,'fake_stage',side_effect=stage): self.run_one()

    def test_ml_name_variants_reject_duplicates(self):
        folder = self.root/'ml';folder.mkdir()
        for level in ('l0c','l0d'):
            (folder/f'awe_{level}_q20_2023335T0329_00142_v02.nc').touch()
        with self.assertRaises(ValueError): r.ml_output(folder,'l0d','00142')

    def test_existing_l0d_lookup_accepts_l0c_name(self):
        folder = self.root/'ml';folder.mkdir()
        f = folder/'awe_l0c_q20_2023335T0329_00142_v02.nc';f.touch()
        self.assertEqual(r.ml_output(folder,'l0d','00142','2023335T0329','v02*'),f)
        with self.assertRaises(ValueError): r.ml_output(folder,'l0d','00142','2023335T0329','v03*')

    def test_l0d_default_cleanup_after_flatfield(self):
        original = self.fake_stage
        def stage(command,cwd,logfile):
            if logfile.stem == 'step_3':
                self.assertFalse((self.work/'l0d').exists())
            original(command,cwd,logfile)
        with patch.object(self,'fake_stage',side_effect=stage): self.run_one()
        self.assertFalse(self.permanent['l0d'].exists())

    def test_save_l0d_keeps_generated_product(self):
        self.args.save_l0d = True
        self.run_one()
        self.assertEqual(len(list(self.permanent['l0d'].glob('*.nc'))),1)

    def test_flatfield_failure_preserves_temporary_l0d(self):
        with self.assertRaises(RuntimeError): self.run_one(fail_stage=2)
        self.assertEqual(len(list((self.work/'l0d').glob('*.nc'))),1)

    def test_flatfield_validation_failure_preserves_temporary_l0d(self):
        with self.assertRaises(ValueError): self.run_one(invalid_stage=2)
        self.assertEqual(len(list((self.work/'l0d').glob('*.nc'))),1)

    def test_unsaved_l0d_leaves_previous_file_untouched(self):
        old = self.permanent['l0d']/'awe_l0c_q20_2023335T0329_00142_v02.nc'
        old.parent.mkdir(parents=True)
        old.write_bytes(b'previous L0D')
        self.run_one()
        self.assertEqual(old.read_bytes(),b'previous L0D')


class ContinueTests(unittest.TestCase):
    def run_scenario(self, options=(), error=RuntimeError('injected failure')):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'l0c'/'2023'/'11'
            folder.mkdir(parents=True)
            for orbit in ('00142','00143'):
                for filt in r.FILTERS:
                    (folder/f'awe_l0c_{filt}_2023335T0329_{orbit}_v02.nc').touch()
            calls = []
            def process(orbit,*args):
                calls.append(orbit)
                if orbit == '00142': raise error
            caught = None
            code = None
            with patch.object(r,'SCRIPTS',{s:Path(__file__) for s in range(1,7)}),patch.object(r,'process_one',side_effect=process):
                try:
                    code = r.main(['--soc-root',str(root),'--year','2023','--month','11',*options])
                except BaseException as exc: caught = exc
            summary = next((root/'pipeline_logs').rglob('summary.json'))
            return calls,code,caught,json.loads(summary.read_text())

    def test_continue_default_records_failure_and_runs_next(self):
        calls,code,caught,records = self.run_scenario()
        self.assertEqual(calls,['00142','00143'])
        self.assertEqual(code,1)
        self.assertIsNone(caught)
        self.assertEqual([r['status'] for r in records],['failed','completed'])
        self.assertIn('injected failure',records[0]['error'])

    def test_disable_stops_at_first_failure(self):
        calls,code,caught,records = self.run_scenario(['--no-continue-on-error'])
        self.assertEqual(calls,['00142'])
        self.assertIsInstance(caught,RuntimeError)
        self.assertEqual(records[0]['status'],'failed')

    def test_ctrl_c_stops_even_when_continue_enabled(self):
        calls,code,caught,records = self.run_scenario(['--continue-on-error'],KeyboardInterrupt())
        self.assertEqual(calls,['00142'])
        self.assertIsInstance(caught,KeyboardInterrupt)
        self.assertEqual(records[0]['status'],'interrupted')

    def test_limit_counts_failed_attempts(self):
        calls,code,caught,records = self.run_scenario(['--limit','1'])
        self.assertEqual(calls,['00142'])
        self.assertEqual(code,1)
        self.assertEqual(len(records),1)


class CloudValidationTests(unittest.TestCase):
    def validate(self, cloud_names, empty=False):
        # Exercise the real validation function through a minimal Dataset double;
        # no TensorFlow model or large AWE files are needed for this schema check.
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        var = MagicMock()
        var.size = 0 if empty else 45
        var.ndim = 3
        var.shape = (1,3,15)
        standard = MagicMock()
        standard.size = 1
        standard.ndim = 1
        standard.shape = (1,)
        variables = {n:standard for n in ('Radiance','Latitude','Longitude','Epoch')}
        variables.update({n:var for n in cloud_names})
        ds = MagicMock()
        ds.variables = variables
        ds.__getitem__.side_effect = variables.__getitem__
        ds.__enter__.return_value = ds
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'awe_l1c_q20_2023335T0329_00142_v02.nc').touch()
            checks = r.expected_outputs(4,{'l1c':folder},'00142')
            specification = folder/'validate.json'
            specification.write_text(json.dumps(checks))
            with patch.dict('sys.modules',{'netCDF4':SimpleNamespace(Dataset=lambda *a:ds)}):
                r.validate_child(specification)
        return var

    def test_processed_only_is_accepted_and_read(self):
        var = self.validate(['Processed_MLCloud'])
        self.assertEqual(var.__getitem__.call_count,2)

    def test_legacy_mlcloud_is_accepted_and_read(self):
        var = self.validate(['MLCloud'])
        self.assertEqual(var.__getitem__.call_count,2)

    def test_no_cloud_variable_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing required cloud variable'):
            self.validate([])

    def test_empty_processed_cloud_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'empty Processed_MLCloud'):
            self.validate(['Processed_MLCloud'],empty=True)


if __name__ == '__main__': unittest.main()
