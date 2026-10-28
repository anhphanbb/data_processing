import copy
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from netCDF4 import Dataset
from PIL import Image
import server


class ReviewTests(unittest.TestCase):
    def test_real_http_frames_review_and_export(self):
        with tempfile.TemporaryDirectory(dir=server.HERE) as directory:
            root = Path(directory)
            path = root/'awe_l1r_q20_2023326T0236_00002_v02.nc'
            with Dataset(path, 'w') as ds:
                for name, size in [('time', 3), ('y', 300), ('x', 300)]:
                    ds.createDimension(name, size)
                ds.createVariable('Radiance', 'f4', ('time','y','x'))[:] = np.broadcast_to(np.arange(300)[:,None], (3,300,300))
            old = server.REVIEWS
            server.REVIEWS = root/'reviews'
            http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
            thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
            base = f'http://127.0.0.1:{http.server_port}'
            def call(route, data=None):
                request = Request(base+route, data=json.dumps(data).encode() if data else None,
                                  headers={'Content-Type':'application/json'} if data else {})
                return urlopen(request).read()
            try:
                files = json.loads(call('/api/scan', {'root':str(root)}))
                self.assertEqual(len(files), 1)
                file_id = files[0]['id']; source = 'cloud_intervals_cleaned_filtered_10272025.csv'
                query = f'?id={file_id}&source={source}'
                meta = json.loads(call('/api/open'+query))
                self.assertEqual(len(meta['expert']), 6)
                self.assertTrue(meta['warnings'])  # Expert labels exceed tiny fixture length.
                raw = np.frombuffer(call(f'/api/frames?id={file_id}&start=0'), dtype='<f4').reshape(3,300,300)
                np.testing.assert_array_equal(raw[0,:,0], np.arange(300))
                packed = call(f'/api/frames?id={file_id}&start=0&gzip=1')
                self.assertEqual(gzip.decompress(packed), raw.tobytes())
                self.assertEqual(server.OPEN_ID, file_id)
                tail = np.frombuffer(call(f'/api/frames?id={file_id}&start=2'), dtype='<f4')
                self.assertEqual(tail.size, 90000)
                with self.assertRaises(HTTPError): call(f'/api/frames?id={file_id}&start=3')
                pixels = np.asarray(Image.open(io.BytesIO(call(f'/api/frame?id={file_id}&frame=0&min=0&max=300'))))
                self.assertGreater(pixels[0,0,0], pixels[-1,0,0])  # origin lower
                review = {'id':file_id,'source':source,'revision':0,'status':'validated',
                          'reviewer':'Automated test','alignment_confirmed':False,
                          'intervals':[{'id':'a','start':0,'end':2,'y':2,'kind':'cloud','notes':'test'}]}
                with self.assertRaises(HTTPError): call('/api/save',review)
                review['alignment_confirmed']=True
                saved=json.loads(call('/api/save',review)); self.assertEqual(saved['revision'],1)
                with self.assertRaises(HTTPError): call('/api/save',review)  # stale revision
                reloaded=json.loads(call('/api/open'+query)); self.assertEqual(reloaded['review']['intervals'],review['intervals'])
                rows=list(csv.DictReader(io.StringIO(call('/api/export'+query).decode())))
                self.assertEqual(rows[0]['(7,2)end'],'2')
                review.update(revision=1,status='draft'); review['intervals'][0]['end']=3
                with self.assertRaises(HTTPError): call('/api/save',review)
                review['intervals'][0].update(end=2,kind='uncertain'); review['status']='validated'
                call('/api/save',review)
                with self.assertRaises(HTTPError): call('/api/export'+query)
                request=Request(base+'/api/config',headers={'Origin':'https://external.example'})
                with self.assertRaises(HTTPError): urlopen(request)
                self.assertEqual(len(list((server.REVIEWS/'history').glob('*.json'))),2)
            finally:
                http.shutdown(); http.server_close(); server.REVIEWS=old
                if server.OPEN_DATASET is not None:
                    server.OPEN_DATASET.close(); server.OPEN_DATASET=None; server.OPEN_ID=None


if __name__=='__main__':
    unittest.main()
