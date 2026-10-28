"""Read-only SSH/SFTP input staging; requires OpenSSH and remote python3.

Authentication is noninteractive (SSH key/agent); passwords are never collected.
"""
from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess


@dataclass(frozen=True, order=True)
class RemoteFile:
    name: str
    path: str
    size: int
    mtime_ns: int

    def signature(self):
        return [self.path, self.size, self.mtime_ns]


def add_arguments(parser):
    parser.add_argument('--ssh-host', help='SSH login, e.g. anh@awex2.physics.usu.edu (key/agent required)')
    parser.add_argument('--ssh-root', help='Absolute remote month directory, e.g. /awedata2/temp_v2/tmp_202401')
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--ssh-key', type=Path, help='Optional private-key path; never a password')
    parser.add_argument('--ssh-cache-root', type=Path, help='Temporary raw-input cache; default WORK_ROOT/ssh_inputs')


class SSHSource:
    def __init__(self, host, root, cache, port=22, key=None):
        if not re.fullmatch(r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9.-]*', host):
            raise ValueError('Invalid SSH host/login')
        if not root.startswith('/') or any(c in root for c in '\r\n\x00\\') or '..' in PurePosixPath(root).parts:
            raise ValueError('ssh-root must be an absolute remote directory without parent traversal')
        if not 1 <= port <= 65535: raise ValueError('Invalid SSH port')
        self.host, self.root, self.port = host, root.rstrip('/') or '/', port
        self.key = str(Path(key).expanduser().resolve()) if key else None
        if self.key and not Path(self.key).is_file(): raise FileNotFoundError(self.key)
        self.identity = hashlib.sha256(f'{host}:{port}:{self.root}'.encode()).hexdigest()[:20]
        self.cache = Path(cache).expanduser().resolve() / self.identity

    def options(self):
        opts = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3']
        if self.key: opts += ['-i', self.key]
        return opts

    def query(self, script, payload):
        code = 'import json, os, hashlib\npayload = ' + repr(payload) + '\n' + script
        encoded = base64.b64encode(code.encode()).decode()
        command = 'python3 -c ' + shlex.quote("import base64; exec(base64.b64decode('" + encoded + "'))")
        result = subprocess.run(['ssh', *self.options(), '-p', str(self.port), self.host, command],
                                capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError('SSH read failed. Check key/agent authentication and known_hosts.\n' + result.stderr[-2000:])
        return json.loads(result.stdout)

    def discover(self, pattern, filters, requested, year, month):
        rows = self.query("""rows = []
for e in os.scandir(payload):
    if e.is_file(follow_symlinks=False) and e.name.endswith('.nc'):
        s = e.stat(follow_symlinks=False)
        rows.append(dict(name=e.name, size=s.st_size, mtime_ns=s.st_mtime_ns))
print(json.dumps(rows))
""", self.root)
        return self.group_rows(rows, pattern, filters, requested, year, month)

    def group_rows(self, rows, pattern, filters, requested, year, month):
        from datetime import datetime
        groups = {}
        for row in rows:
            name = row['name']
            m = pattern.fullmatch(name)
            if not m: continue
            filt, stamp, orbit, version, _ = m.groups()
            if version.lower() != 'v02' or (requested and orbit not in requested): continue
            date = datetime.strptime(stamp, '%Y%jT%H%M')
            if (date.year, date.month) != (year, month): continue
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
                raise ValueError('Unsafe remote filename')
            group = groups.setdefault(orbit, {})
            if filt.lower() in group: raise ValueError(f'Orbit {orbit}: duplicate {filt}')
            if int(row['size']) <= 0: raise ValueError(f'Empty remote file: {name}')
            group[filt.lower()] = RemoteFile(name, str(PurePosixPath(self.root)/name),
                                             int(row['size']), int(row['mtime_ns']))
        for orbit, group in groups.items():
            if set(group) != set(filters): raise ValueError(f'Orbit {orbit}: incomplete remote filter set')
            if len({pattern.fullmatch(f.name).groups()[1:4] for f in group.values()}) != 1:
                raise ValueError(f'Orbit {orbit}: timestamps/versions differ')
        if not groups: raise FileNotFoundError('No complete v02 orbits match this remote month/selection')
        return dict(sorted(groups.items()))

    def checksum(self, source):
        return self.query("""p, size, stamp = payload
before = os.stat(p)
if (before.st_size, before.st_mtime_ns) != (size, stamp):
    raise RuntimeError('Remote file changed since discovery: ' + p)
h = hashlib.sha256()
with open(p, 'rb') as f:
    for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
after = os.stat(p)
if (after.st_size, after.st_mtime_ns) != (size, stamp):
    raise RuntimeError('Remote file changed while hashing: ' + p)
print(json.dumps(h.hexdigest()))
""", source.signature())

    def folder(self, orbit):
        if not re.fullmatch(r'\d{5}', orbit): raise ValueError('Invalid orbit')
        folder = self.cache/orbit
        if not folder.resolve().is_relative_to(self.cache): raise ValueError('Cache path escapes source directory')
        return folder

    @staticmethod
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def quote_sftp(value):
        return '"' + str(value).replace('\\', '/').replace('"', '\\"') + '"'

    def stage(self, orbit, group, run_logged, logdir):
        folder = self.folder(orbit); folder.mkdir(parents=True, exist_ok=True)
        # A leftover lock after a killed process needs explicit operator review.
        lock = folder/'.download.lock'
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        try:
            result = {}
            for filt, source in group.items():
                target = folder/source.name
                part = target.with_suffix(target.suffix+'.partial')
                metadata = target.with_suffix(target.suffix+'.source.json')
                expected = dict(source=source.signature(), sha256=self.checksum(source))
                old = json.loads(metadata.read_text()) if metadata.exists() else None
                if old != expected:
                    part.unlink(missing_ok=True); target.unlink(missing_ok=True)
                metadata.write_text(json.dumps(expected))
                if target.exists():
                    if target.stat().st_size == source.size and self.digest(target) == expected['sha256']:
                        result[filt] = target; continue
                    target.unlink()
                if part.exists() and part.stat().st_size > source.size: part.unlink()
                if part.exists() and part.stat().st_size == source.size:
                    if self.digest(part) == expected['sha256']:
                        os.replace(part, target); result[filt] = target; continue
                    part.unlink()
                batch = folder/(filt+'.sftp')
                batch.write_text('reget ' + self.quote_sftp(source.path) + ' ' + self.quote_sftp(part) + '\n')
                try:
                    run_logged(['sftp', *self.options(), '-P', str(self.port), '-b', str(batch), self.host],
                               folder, logdir/('download_'+filt+'.log'))
                finally:
                    batch.unlink(missing_ok=True)
                if part.stat().st_size != source.size or self.digest(part) != expected['sha256']:
                    part.unlink(missing_ok=True)
                    raise ValueError(f'Download checksum/size mismatch: {source.name}')
                os.replace(part, target)
                result[filt] = target
            return result
        finally:
            lock.unlink(missing_ok=True)

    def cleanup(self, orbit, group):
        folder = self.folder(orbit)
        if (folder/'.download.lock').exists(): raise RuntimeError('Refusing cleanup during download')
        for source in group.values():
            for suffix in ('', '.partial', '.source.json'):
                path = folder/(source.name+suffix)
                if not path.resolve().is_relative_to(self.cache): raise ValueError('Unsafe cleanup path')
                path.unlink(missing_ok=True)
        if folder.exists() and not any(folder.iterdir()): folder.rmdir()
