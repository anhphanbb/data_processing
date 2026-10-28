"""Local-only MLCloud review application. Python 3.10, numpy, netCDF4, Pillow."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import webbrowser

import numpy as np
from netCDF4 import Dataset
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LABELS = ROOT/'csv'/'mlcloud'
REVIEWS = HERE/'reviews'
LOCK = threading.RLock()  # netCDF/HDF5 calls must not run concurrently.
FILES = {}
OPEN_DATASET = None
OPEN_ID = None


def dataset(file_id):
    """Reuse one read-only file handle; callers hold LOCK."""
    global OPEN_DATASET, OPEN_ID
    if OPEN_ID != file_id:
        if OPEN_DATASET is not None:
            OPEN_DATASET.close()
        OPEN_DATASET = Dataset(FILES[file_id])
        OPEN_ID = file_id
    return OPEN_DATASET
PATTERN = re.compile(r'^awe_l1[rc]_q20_(\d{7}T\d{4})_(\d{5})_.*\.nc$')
SOURCES = {p.name: p for p in LABELS.glob('*.csv')}


def identity(path):
    stat = path.stat()
    return hashlib.sha256(f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:24]


def load_expert(source, orbit):
    result, warnings = [], []
    with SOURCES[source].open(encoding='utf-8-sig', newline='') as stream:
        for number, row in enumerate(csv.DictReader(stream), 2):
            try:
                if int(float(row.get('Orbit #') or '-1')) != orbit:
                    continue
            except ValueError:
                continue
            for y, region in ((2, 'Top'), (1, 'Middle'), (0, 'Bottom')):
                start = row.get(f'(7,{y})start', row.get(f'Start ({region})', ''))
                end = row.get(f'(7,{y})end', row.get(f'End ({region})', ''))
                if not start and not end:
                    continue
                try:
                    a, b = float(start), float(end)
                    if not a.is_integer() or not b.is_integer() or a < 0 or b < a:
                        raise ValueError()
                except (ValueError, TypeError):
                    warnings.append(f'CSV row {number}: invalid {region.lower()} interval {start}–{end}')
                    continue
                lights = (row.get('Ground Lights') or '').strip()
                result.append({'id': f'expert-{number}-{y}', 'y': y, 'start': int(a), 'end': int(b),
                               'kind': 'ground_light' if lights else 'cloud',
                               'notes': row.get('Notes') or '', 'source_row': number})
    return result, warnings


def metadata(file_id, source):
    path = FILES[file_id]
    match = PATTERN.fullmatch(path.name)
    with LOCK, Dataset(path) as ds:
        var = ds['Radiance']
        if var.ndim != 3 or var.shape[1:] != (300, 300):
            raise ValueError('MLCloud review requires Radiance(time, 300, 300) in an L1R/L1C q20 file.')
        count = var.shape[0]
        units = getattr(var, 'units', 'Radiance')
        positions = {}
        for key in ('ISS_Latitude', 'ISS_Longitude'):
            if key in ds.variables:
                values = np.ma.filled(ds[key][:].astype('f8'), np.nan).reshape(-1)
                positions[key] = [float(v) if np.isfinite(v) else None for v in values]
    expert, warnings = load_expert(source, int(match[2]))
    if any(i['end'] >= count for i in expert):
        warnings.append('Some expert intervals exceed this file’s frame count. Check source alignment.')
    review_path = REVIEWS/f'{file_id}-{hashlib.sha256(source.encode()).hexdigest()[:8]}.json'
    review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else None
    return {'id': file_id, 'name': path.name, 'path': str(path), 'orbit': int(match[2]),
            'frames': count, 'units': units, 'source': source, 'expert': expert,
            'positions': positions, 'warnings': warnings, 'review': review, 'revision': review['revision'] if review else 0}


def validate_review(data, count):
    if data.get('status') not in ('draft', 'validated'):
        raise ValueError('Unknown review status')
    if data['status'] == 'validated' and (not data.get('alignment_confirmed') or not data.get('reviewer', '').strip()):
        raise ValueError('Confirm frame alignment and enter a reviewer name before validating.')
    intervals = data.get('intervals')
    if not isinstance(intervals, list) or len(intervals) > 10000:
        raise ValueError('Invalid interval list')
    for item in intervals:
        a, b, y = item.get('start'), item.get('end'), item.get('y')
        if any(type(v) is not int for v in (a, b, y)) or not 0 <= a <= b < count or y not in (0, 1, 2):
            raise ValueError('Intervals must have integer bounds within this orbit and a valid box row.')
        if item.get('kind') not in ('cloud', 'ground_light', 'uncertain'):
            raise ValueError('Unknown label type')
        if len(str(item.get('notes', ''))) > 4000:
            raise ValueError('Interval note is too long')


def save_review(data):
    with LOCK:
        meta = metadata(data['id'], data['source'])
        validate_review(data, meta['frames'])
        if data.get('revision') != meta['revision']:
            raise ValueError('This review changed in another window. Reopen the orbit before saving.')
        data = {k: data.get(k) for k in ('id', 'source', 'intervals', 'status', 'reviewer', 'alignment_confirmed')}
        data.update(revision=meta['revision']+1, file=meta['path'], frames=meta['frames'],
                    orbit=meta['orbit'], updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    source_sha256=hashlib.sha256(SOURCES[data['source']].read_bytes()).hexdigest())
        REVIEWS.mkdir(exist_ok=True)
        path = REVIEWS/f"{data['id']}-{hashlib.sha256(data['source'].encode()).hexdigest()[:8]}.json"
        # Retain each revision for recovery as well as the current review.
        history = REVIEWS/'history'
        history.mkdir(exist_ok=True)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        (history/f'{path.stem}-r{data["revision"]}.json').write_text(text, encoding='utf-8')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
        return data


class Handler(BaseHTTPRequestHandler):
    def send(self, content, content_type='application/json', status=200):
        if isinstance(content, (dict, list)):
            content = json.dumps(content, allow_nan=False).encode()
        if isinstance(content, str):
            content = content.encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(content)

    def allowed(self):
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        expected = f'127.0.0.1:{self.server.server_port}'
        return host in (expected, f'localhost:{self.server.server_port}') and (not origin or origin == 'http://'+host)

    def do_GET(self):
        if not self.allowed():
            return self.send({'error': 'Local requests only'}, status=403)
        try:
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == '/api/config':
                return self.send({'sources': sorted(SOURCES), 'defaultRoot': str(ROOT/'validation_outputs'/'orbit02'),
                                  'reviewFolder': str(REVIEWS)})
            if url.path == '/api/open':
                return self.send(metadata(query['id'], query['source']))
            if url.path == '/api/frames':
                start = int(query['start'])
                with LOCK:
                    ds = dataset(query['id'])
                    count = ds['Radiance'].shape[0]
                    if not 0 <= start < count:
                        raise ValueError('Frame outside orbit')
                    stop = min(start + 16, count)
                    raw = np.ma.filled(ds['Radiance'][start:stop].astype('f4'), np.nan)
                    content = np.asarray(raw, dtype='<f4').tobytes()
                if query.get('gzip') == '1':
                    content = gzip.compress(content, compresslevel=1)
                return self.send(content, 'application/octet-stream')
            if url.path == '/api/frame':
                frame = int(query['frame']); lo = float(query.get('min', 0)); hi = float(query.get('max', 20))
                if not all(math.isfinite(x) for x in (lo, hi)) or hi <= lo:
                    raise ValueError('Maximum radiance must exceed minimum.')
                with LOCK, Dataset(FILES[query['id']]) as ds:
                    if not 0 <= frame < ds['Radiance'].shape[0]:
                        raise ValueError('Frame outside orbit')
                    raw = np.ma.filled(ds['Radiance'][frame].astype('f4'), np.nan)
                valid = np.isfinite(raw)
                grey = (np.clip((np.nan_to_num(raw, nan=lo)-lo)/(hi-lo), 0, 1)*255).astype('uint8')
                rgb = np.repeat(grey[:, :, None], 3, axis=2)
                rgb[~valid] = [77, 46, 102]
                stream = io.BytesIO(); Image.fromarray(rgb[::-1]).save(stream, format='PNG')
                return self.send(stream.getvalue(), 'image/png')
            if url.path == '/api/export':
                meta = metadata(query['id'], query['source']); review = meta['review']
                if not review or review['status'] != 'validated':
                    raise ValueError('Save a validated review before exporting training labels.')
                if any(i['kind'] != 'cloud' for i in review['intervals']):
                    raise ValueError('Ground-light/uncertain labels need a separate training policy. Resolve them before exporting cloud-only training CSV.')
                out = io.StringIO(newline='')
                fields = ['Orbit #']+[f'(7,{y}){side}' for y in (2, 1, 0) for side in ('start', 'end')]+['Ground Lights','Notes']
                writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
                for item in review['intervals']:
                    writer.writerow({'Orbit #': meta['orbit'], f'(7,{item["y"]})start':item['start'],
                                     f'(7,{item["y"]})end':item['end'], 'Notes':item['notes']})
                return self.send(out.getvalue(), 'text/csv; charset=utf-8')
            static = {'/': 'index.html', '/app.js': 'app.js', '/orbit-loader.js': 'orbit-loader.js', '/style.css': 'style.css'}
            if url.path in static:
                types = {'/':'text/html; charset=utf-8', '/app.js':'text/javascript', '/orbit-loader.js':'text/javascript', '/style.css':'text/css'}
                return self.send((HERE/static[url.path]).read_bytes(), types[url.path])
            self.send({'error':'Not found'}, status=404)
        except (ValueError, KeyError, OSError) as exc:
            self.send({'error': str(exc)}, status=400)

    def do_POST(self):
        if not self.allowed() or self.headers.get('Content-Type') != 'application/json':
            return self.send({'error':'Local JSON requests only'}, status=403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 2_000_000:
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(length))
            if self.path == '/api/scan':
                directory = Path(data['root']).expanduser().resolve()
                if not directory.is_dir():
                    raise ValueError('Folder does not exist or is unavailable.')
                found = []
                for path in directory.rglob('*.nc'):
                    match = PATTERN.fullmatch(path.name)
                    if not match:
                        continue
                    key = identity(path); FILES[key] = path
                    found.append({'id':key, 'name':path.name, 'orbit':int(match[2]), 'stamp':match[1]})
                    if len(found) >= 1000:
                        break
                return self.send(sorted(found, key=lambda x:(x['stamp'], x['name'])))
            if self.path == '/api/save':
                return self.send(save_review(data))
            self.send({'error':'Not found'}, status=404)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            self.send({'error':str(exc)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{args.port}'
    print(f'MLCloud Review: {url}', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
