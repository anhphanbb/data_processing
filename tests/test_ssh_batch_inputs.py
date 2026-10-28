"""Offline SSH staging and batch integration tests; no server credentials needed."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ssh_batch_inputs as ssh
import run_awe_pipeline_batch as batch


class SSHInputTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'validation_outputs'; parent.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = ssh.SSHSource('anh@example.edu', '/awedata2/temp_v2/tmp_202401', self.root/'cache')
        self.body = b'example netcdf bytes'
        self.file = ssh.RemoteFile('awe_l0c_q20_2024001T0127_00621_v02.nc',
                                  '/awedata2/temp_v2/tmp_202401/awe_l0c_q20_2024001T0127_00621_v02.nc',
                                  len(self.body), 123)
        self.group = {'q20': self.file}
        self.logs = self.root/'logs'; self.logs.mkdir()
        self.hash = hashlib.sha256(self.body).hexdigest()
        self.addCleanup(patch.stopall)
        patch.object(self.source, 'checksum', return_value=self.hash).start()

    def rows(self):
        return [dict(name=f'awe_l0c_{f}_2024001T0127_00621_v{v}.nc', size=12, mtime_ns=123)
                for f in batch.core.FILTERS for v in ('02', '22')]

    def groups(self, rows):
        return self.source.group_rows(rows, batch.core.INPUT_RE, batch.core.FILTERS, None, 2024, 1)

    def transfer(self, command, cwd, log):
        self.assertEqual(command[0], 'sftp')
        self.assertIn('BatchMode=yes', command)
        self.assertIn('StrictHostKeyChecking=yes', command)
        self.assertIn('reget ', Path(command[command.index('-b')+1]).read_text())
        (cwd/(self.file.name+'.partial')).write_bytes(self.body)

    def test_inventory_filters_version_and_month(self):
        rows = self.rows() + [dict(name='awe_l0c_q20_2024032T0127_00622_v02.nc',size=12,mtime_ns=1)]
        groups = self.groups(rows)
        self.assertEqual(list(groups), ['00621'])
        self.assertEqual(set(groups['00621']), set(batch.core.FILTERS))
        self.assertTrue(all(f.name.endswith('_v02.nc') for f in groups['00621'].values()))

    def test_rejects_incomplete_duplicate_and_mismatched_time(self):
        rows = [r for r in self.rows() if '_v02' in r['name']]
        for invalid in (rows[:-1], rows+[rows[0]],
                        [dict(rows[0], name=rows[0]['name'].replace('T0127','T0128'))]+rows[1:]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): self.groups(invalid)

    def test_safe_endpoints(self):
        for host,root in [('-oProxyCommand=bad','/data'),('a@b','relative'),('a@b','/data/../etc')]:
            with self.assertRaises(ValueError): ssh.SSHSource(host,root,self.root)

    def test_stage_verified_reuse_and_cleanup(self):
        paths = self.source.stage('00621',self.group,self.transfer,self.logs)
        self.assertEqual(paths['q20'].read_bytes(),self.body)
        def no_transfer(*args): self.fail('Verified cache should not download again')
        self.source.stage('00621',self.group,no_transfer,self.logs)
        self.source.cleanup('00621',self.group)
        self.assertFalse(paths['q20'].exists())

    def test_interrupted_transfer_resumes(self):
        def interrupt(command,cwd,log):
            (cwd/(self.file.name+'.partial')).write_bytes(self.body[:4])
            raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError): self.source.stage('00621',self.group,interrupt,self.logs)
        part=self.source.folder('00621')/(self.file.name+'.partial')
        self.assertEqual(part.read_bytes(),self.body[:4])
        def resume(command,cwd,log):
            self.assertEqual(part.read_bytes(),self.body[:4]); self.transfer(command,cwd,log)
        self.source.stage('00621',self.group,resume,self.logs)
        self.assertFalse(part.exists())

    def test_bad_checksum_never_publishes(self):
        def corrupt(command,cwd,log): (cwd/(self.file.name+'.partial')).write_bytes(b'x'*self.file.size)
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.source.stage('00621',self.group,corrupt,self.logs)
        self.assertFalse((self.source.folder('00621')/self.file.name).exists())

    def test_changed_manifest_discards_partial(self):
        folder=self.source.folder('00621');folder.mkdir(parents=True)
        part=folder/(self.file.name+'.partial');part.write_bytes(b'wrong')
        (folder/(self.file.name+'.source.json')).write_text('{}')
        def transfer(command,cwd,log):
            self.assertFalse(part.exists()); self.transfer(command,cwd,log)
        self.source.stage('00621',self.group,transfer,self.logs)

    def test_cleanup_only_after_receipt(self):
        paths={level:self.root/'work'/level for level in batch.core.LEVELS}
        for p in paths.values():p.mkdir(parents=True)
        (paths['l1r']/'awe_l1r_q20_2024001T0127_00621_v02.nc').write_bytes(b'output')
        permanent={level:self.root/'saved'/level for level in batch.core.LEVELS}
        permanent['master']=permanent['l1r']
        job=dict(orbit='00621',signature='test',group=self.group,receipt=self.root/'receipt.json')
        args=SimpleNamespace(start_step=0,save_l0c=False,save_l0d=False,save_l1f=False,
                             save_l1r=True,overwrite=True,ssh_source=self.source)
        with patch.object(self.source,'cleanup') as clean, patch.object(batch.core,'publish_file',side_effect=OSError('upload failed')):
            with self.assertRaises(OSError):batch.save_products(paths,permanent,job,args,self.logs)
            clean.assert_not_called()
        def clean(orbit,group): self.assertTrue(job['receipt'].is_file())
        with patch.object(self.source,'cleanup',side_effect=clean) as cleanup:
            batch.save_products(paths,permanent,job,args,self.logs)
            cleanup.assert_called_once()

    def test_remote_input_stage_precedes_glare(self):
        paths={'l0c':self.root/'processed'};paths['l0c'].mkdir()
        args=SimpleNamespace(start_step=0,ssh_source=self.source,workers=1,glare_csv=self.root/'glare.csv')
        records=[]
        def run(command,cwd,log):
            if command[0]=='sftp': self.transfer(command,cwd,log)
            else:
                index=command.index('--files')
                self.assertTrue(all(Path(p).is_file() for p in command[index+1:]))
        # Supply four filters for the normal step-0 command.
        with patch.object(self.source,'stage',return_value={f:self.logs/'input.nc' for f in batch.core.FILTERS}) as stage:
            (self.logs/'input.nc').write_bytes(self.body)
            with patch.object(batch.core,'run_logged',side_effect=run), patch.object(batch.core,'validate_outputs'):
                batch.core.prepare_l0c('00621',self.group,paths,args,self.logs,records)
            stage.assert_called_once();self.assertEqual(records[0]['step'],0)

    def test_dry_run_inventories_without_download(self):
        groups=self.groups(self.rows())
        with patch.object(ssh.SSHSource,'discover',return_value=groups), patch.object(ssh.SSHSource,'stage') as stage:
            code=batch.main(['--year','2024','--month','1','--start-step','0','--dry-run',
                             '--no-save-to-server','--soc-root',str(self.root/'soc'),
                             '--work-root',str(self.root/'work'), '--ssh-host','anh@example.edu',
                             '--ssh-root','/awedata2/temp_v2/tmp_202401'])
            self.assertEqual(code,0);stage.assert_not_called()

    def test_remote_fingerprint_requires_no_local_inputs(self):
        group=self.groups(self.rows())['00621']
        args=SimpleNamespace(start_step=0,ssh_source=self.source,save_l0c=False,save_l0d=False,
                             save_l1f=False,save_l1r=True,workers=1,soc_root=self.root/'soc',
                             year=2024,month=1,save_to_server=False,glare_csv=ROOT/'csv/all_months.csv')
        first=batch.core.fingerprint(group,args)
        self.assertEqual(first,batch.core.fingerprint(group,args))
        args.ssh_source=ssh.SSHSource('other@example.edu',self.source.root,self.root/'cache')
        self.assertNotEqual(first,batch.core.fingerprint(group,args))

    def test_remote_python_protocol_and_source_change(self):
        # Execute the exact SSH payload with a local Python in place of a server.
        actual_run=subprocess.run
        remote=ssh.SSHSource('anh@example.edu','/data',self.root/'other_cache')
        file=self.root/'payload.nc';file.write_bytes(self.body)
        stat=file.stat()
        source=ssh.RemoteFile(file.name,str(file),stat.st_size,stat.st_mtime_ns)
        def fake_ssh(command,**kwargs):
            self.assertEqual(command[0],'ssh')
            self.assertIn('StrictHostKeyChecking=yes',command)
            payload=shlex.split(command[-1])
            self.assertEqual(payload[:2],['python3','-c'])
            return actual_run([sys.executable,'-c',payload[2]],**kwargs)
        with patch.object(ssh.subprocess,'run',side_effect=fake_ssh):
            self.assertEqual(remote.checksum(source),self.hash)
            file.write_bytes(self.body+b'changed')
            with self.assertRaisesRegex(RuntimeError,'SSH read failed'):remote.checksum(source)

    def test_local_input_dry_run_unchanged(self):
        source=self.root/'soc/l0c/2024/01';source.mkdir(parents=True)
        for row in self.rows(): (source/row['name']).touch()
        with patch.object(ssh.SSHSource,'discover') as discover:
            code=batch.main(['--year','2024','--month','1','--start-step','1','--dry-run',
                             '--no-save-to-server','--soc-root',str(self.root/'soc'),
                             '--work-root',str(self.root/'work')])
            self.assertEqual(code,0);discover.assert_not_called()


if __name__=='__main__': unittest.main()
