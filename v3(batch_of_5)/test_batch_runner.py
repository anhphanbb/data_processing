"""Batch scheduling and retention tests; scientific stages use tiny stand-ins."""
import argparse
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import run_awe_pipeline_batch as b


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root/'work'
        self.logs = self.root/'logs'
        self.permanent = b.core.month_paths(self.root/'soc',2023,11)
        self.args = argparse.Namespace(workers=20,orbits_at_a_time=5,start_step=1,
            save_l0d=False,save_l1f=False,save_l1r=False,continue_on_error=True,overwrite=False)
        self.jobs = []
        for n in range(1,6):
            orbit = f'{n:05d}'
            group = {}
            for filt in b.core.FILTERS:
                p = self.permanent['l0c']/f'awe_l0c_{filt}_2023326T0103_{orbit}_v02.nc'
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b'original')
                group[filt] = p
            self.jobs.append({'orbit':orbit,'group':group,'record':{'orbit':orbit,'status':'pending'},
                             'signature':'test','receipt':self.root/'receipts'/f'{orbit}.json'})
        self.calls = []
        self.fail_orbit = None
        self.fail_step = None
        self.interrupt = False
        self.exit_after_outputs = False

    def fake_run(self,command,cwd,logfile):
        step = int(logfile.stem.split('_')[1])
        # Infer the actual source directory passed to the child, including retries.
        if step in (1,4): source = Path(command[5])
        elif step in (2,3): source = Path(command[3])
        elif step == 5: source = Path(command[command.index('--l1r-input')+1])
        else: source = Path(command[command.index('--l1a-input')+1])
        paths = b.paths_in(source.parent)
        orbits = sorted({p.name.split('_')[4] for p in source.glob('awe_*_q20_*.nc')})
        self.calls.append((step,orbits,command))
        if self.interrupt and step == 2: raise KeyboardInterrupt()
        if step == 3 and not self.args.save_l0d: self.assertFalse(list(paths['l0d'].glob('*.nc')))
        if step == 4 and not self.args.save_l1f: self.assertFalse(list(paths['l1f'].glob('*.nc')))
        if step == 6 and not self.args.save_l1r: self.assertFalse(list(paths['l1r'].glob('*.nc')))
        failed = False
        for orbit in orbits:
            if orbit == self.fail_orbit and step == self.fail_step:
                failed = True
                # Leave an incomplete output to verify it is isolated.
                if step == 3:
                    paths['l1r'].mkdir(parents=True,exist_ok=True)
                    (paths['l1r']/f'awe_l1r_q20_2023326T0103_{orbit}_v02.nc.partial').write_bytes(b'partial')
                continue
            levels = {1:['l0d'],2:['l1f'],3:['l1r'],4:['l1c'],5:['l1a'],6:['l2a','l3a','l3c']}[step]
            for level in levels:
                filters = b.core.FILTERS if level in ('l1f','l1r','l1a') else ('tmp',) if level in ('l2a','l3a') else ('q20',)
                folders = [paths[level]/f'{x}x{x}' for x in b.RESOLUTIONS] if level in ('l2a','l3a') else [paths[level]]
                for folder in folders:
                    folder.mkdir(parents=True,exist_ok=True)
                    name_level = 'l0c' if level == 'l0d' else level
                    for filt in filters:
                        (folder/f'awe_{name_level}_{filt}_2023326T0103_{orbit}_v02.nc').write_bytes(b'product')
            if step == 3:
                (paths['l1r']/f'Master{orbit}.mgf').write_bytes(struct.pack('<ii',1,1)+bytes(16))
        if failed or (self.exit_after_outputs and step == 3 and len(orbits)>1):
            raise RuntimeError('injected child error')

    def run_batch(self):
        with patch.object(b.core,'run_logged',side_effect=self.fake_run),patch.object(
                b.core,'validate_outputs',side_effect=lambda step,paths,orbit,logs:b.core.expected_outputs(step,paths,orbit)):
            b.process_batch(self.jobs,self.args,self.permanent,self.work,self.logs,lambda:None)

    def test_five_orbits_six_stage_calls_twenty_workers_and_cleanup(self):
        self.run_batch()
        self.assertEqual([x[0] for x in self.calls],[1,2,3,4,5,6])
        self.assertTrue(all(len(x[1])==5 for x in self.calls))
        command = self.calls[1][2]
        self.assertEqual(command[command.index('--orbits-at-a-time')+1],'5')
        self.assertEqual(command[command.index('--workers')+1],'20')
        self.assertFalse(self.work.exists())
        self.assertEqual(len(list(self.permanent['l1r'].glob('*.mgf'))),5)
        for level in ('l0d','l1f'):
            self.assertFalse(self.permanent[level].exists())
        self.assertEqual(len(list(self.permanent['l1r'].glob('*.nc'))),0)
        for job in self.jobs:
            self.assertEqual(job['record']['status'],'completed')
            self.assertTrue(b.core.receipt_matches(job['receipt'],'test'))
            for src in job['group'].values(): self.assertEqual(src.read_bytes(),b'original')

    def test_failed_orbit_isolated_others_finish(self):
        self.fail_orbit,self.fail_step = '00002',3
        self.run_batch()
        self.assertEqual([j['record']['status'] for j in self.jobs],['completed','failed','completed','completed','completed'])
        retained = b.paths_in(self.work/'failed'/'00002')
        self.assertEqual(len(list(retained['l1f'].glob('*.nc'))),4)
        self.assertTrue(list(retained['l1r'].glob('*.partial')))
        self.assertFalse(self.jobs[1]['receipt'].exists())
        later_calls = [orbits for step,orbits,_ in self.calls if step>3]
        self.assertTrue(all('00002' not in orbits for orbits in later_calls))
        self.assertFalse(list((self.work/'active').rglob('*.nc')))

    def test_nonzero_exit_with_all_outputs_requires_successful_retries(self):
        self.exit_after_outputs = True
        self.run_batch()
        self.assertEqual(sum(step==3 for step,_,_ in self.calls),6)
        self.assertTrue(all(j['record']['status']=='completed' for j in self.jobs))
        self.assertTrue(all(j['record']['retries'][0]['step']==3 for j in self.jobs))

    def test_stage_six_failure_does_not_publish_incomplete_orbit(self):
        self.fail_orbit,self.fail_step = '00002',6
        self.run_batch()
        self.assertEqual(self.jobs[1]['record']['status'],'failed')
        self.assertFalse(list(self.permanent['l3c'].glob('*_00002_*.nc')))
        self.assertTrue((self.work/'failed'/'00002'/'l1r'/'Master00002.mgf').is_file())
        self.assertEqual(len(list(self.permanent['l3c'].glob('*.nc'))),4)

    def test_no_continue_stops_on_batch_error(self):
        self.args.continue_on_error = False
        self.fail_orbit,self.fail_step = '00002',3
        with self.assertRaises(RuntimeError): self.run_batch()
        self.assertEqual([x[0] for x in self.calls],[1,2,3])
        self.assertTrue(self.work.exists())
        self.assertTrue(all(j['record']['status']=='aborted' for j in self.jobs))

    def test_ctrl_c_stops_all(self):
        self.interrupt = True
        with self.assertRaises(KeyboardInterrupt): self.run_batch()
        self.assertEqual([x[0] for x in self.calls],[1,2])
        self.assertTrue(all(j['record']['status']=='interrupted' for j in self.jobs))

    def test_retention_flags_keep_all_generated_levels(self):
        self.args.save_l0d = self.args.save_l1f = self.args.save_l1r = True
        self.run_batch()
        self.assertEqual(len(list(self.permanent['l0d'].glob('*.nc'))),5)
        self.assertEqual(len(list(self.permanent['l1f'].glob('*.nc'))),20)
        self.assertEqual(len(list(self.permanent['l1r'].glob('*.nc'))),20)

    def test_start_two_preserves_original_l0d(self):
        self.args.start_step = 2
        for job in self.jobs:
            p = self.permanent['l0d']/job['group']['q20'].name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b'existing MLSPB')
            job['group']['l0d'] = p
        self.run_batch()
        self.assertEqual([x[0] for x in self.calls],[2,3,4,5,6])
        for job in self.jobs: self.assertEqual(job['group']['l0d'].read_bytes(),b'existing MLSPB')

    def test_publish_failure_keeps_orbit_and_continues(self):
        real = b.core.publish_file
        def publish(src,dst):
            if '_00002_' in dst.name: raise OSError('injected output failure')
            real(src,dst)
        with patch.object(b.core,'publish_file',side_effect=publish): self.run_batch()
        self.assertEqual(self.jobs[1]['record']['status'],'failed')
        self.assertTrue((self.work/'failed'/'00002'/'l3c').exists())
        self.assertEqual(self.jobs[-1]['record']['status'],'completed')

    def test_main_batches_are_bounded_and_final_batch_is_smaller(self):
        batches = []
        def process(jobs,args,permanent,work,logs,checkpoint):
            batches.append([j['orbit'] for j in jobs])
            for job in jobs: job['record']['status']='completed'
            checkpoint()
        with patch.object(b,'SCRIPTS',{n:Path(__file__) for n in range(1,7)}),patch.object(b,'process_batch',side_effect=process):
            code = b.main(['--soc-root',str(self.root/'soc'),'--batch-size','3','--limit','5'])
        self.assertEqual(code,0)
        self.assertEqual(batches,[['00001','00002','00003'],['00004','00005']])

    def test_dry_run_is_read_only(self):
        before = set(self.root.rglob('*'))
        code = b.main(['--soc-root',str(self.root/'soc'),'--dry-run','--batch-size','3'])
        self.assertEqual(code,0)
        self.assertEqual(set(self.root.rglob('*')),before)


if __name__ == '__main__': unittest.main()
