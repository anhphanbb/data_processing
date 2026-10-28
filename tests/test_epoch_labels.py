import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_cloud_labels_to_epoch import convert
from batch_convert_cloud_labels_to_epoch import batch_convert


class EpochLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'awe_l1r_q20_2023336T0000_00002_v01.nc'
        self.epochs = np.array([1356566418.125, 1356566419.25, 1356566421.5, 1356566422.75])
        with Dataset(self.source, 'w') as ds:
            ds.createDimension('time', 4)
            epoch = ds.createVariable('Epoch', 'f8', ('time',))
            epoch.units = 'seconds since 1980-01-06'
            epoch[:] = self.epochs
            ds.createVariable('Radiance', 'f4', ('time',))[:] = 0
        self.labels = self.root / 'labels.csv'
        self.labels.write_text('Orbit #,Start (Top),End (Top),Start (Middle),End (Middle),Start (Bottom),End (Bottom)\n2,1,2,,,0,0\n2,2,3,,,,\n')

    def test_nc_and_csv_roundtrip(self):
        nc, csv_path = self.root / 'out.nc', self.root / 'out.csv'
        convert(self.source, self.labels, nc)
        with Dataset(nc) as ds:
            np.testing.assert_array_equal(ds['epoch'][:], self.epochs)
            np.testing.assert_array_equal(ds['cloud'][:], [[0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
            self.assertEqual(ds['epoch'].units, 'seconds since 1980-01-06')
            self.assertEqual(ds['orbit'][...], 2)
        convert(self.source, self.labels, csv_path)
        with csv_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        np.testing.assert_array_equal([float(r['epoch']) for r in rows[::3]], self.epochs)

    def test_training_columns_and_single_frame(self):
        self.labels.write_text('Orbit #,"(7,0)start","(7,0)end"\n2,3,3\n')
        out = self.root / 'training.nc'
        convert(self.source, self.labels, out)
        with Dataset(out) as ds:
            np.testing.assert_array_equal(ds['cloud'][:, 0], [0, 0, 0, 1])

    def test_reject_bad_intervals_and_ground_lights(self):
        for start, end, lights in [('1', '4', ''), ('1.5', '2', ''), ('1', '', ''), ('2', '1', ''), ('0', '1', 'Ground Lights')]:
            with self.subTest(start=start, end=end, lights=lights):
                self.labels.write_text(f'Orbit #,"(7,0)start","(7,0)end",Ground Lights\n2,{start},{end},{lights}\n')
                with self.assertRaises(ValueError):
                    convert(self.source, self.labels, self.root / 'bad.nc')
                self.assertFalse((self.root / 'bad.nc').exists())

    def test_absent_orbit_requires_explicit_policy(self):
        self.labels.write_text('Orbit #,"(7,0)start","(7,0)end"\n3,0,1\n')
        with self.assertRaises(ValueError):
            convert(self.source, self.labels, self.root / 'missing.nc')
        convert(self.source, self.labels, self.root / 'clear.nc', True)
        with Dataset(self.root / 'clear.nc') as ds:
            self.assertFalse(ds['cloud'][:].any())

    def test_duplicate_epochs_and_overwrite(self):
        out = self.root / 'out.nc'
        out.write_bytes(b'keep')
        with self.assertRaises(ValueError):
            convert(self.source, self.labels, out)
        self.assertEqual(out.read_bytes(), b'keep')
        with Dataset(self.source, 'a') as ds:
            ds['Epoch'][1] = self.epochs[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            convert(self.source, self.labels, self.root / 'duplicate.nc')

    def test_batch_nested_missing_and_ambiguous(self):
        folder = self.root / 'input' / '2023' / '11'
        folder.mkdir(parents=True)
        self.source.rename(folder / self.source.name)
        with self.labels.open('a') as stream:
            stream.write('4,0,1,,,,\n')
        counts, report = batch_convert(self.root / 'input', self.labels, self.root / 'batch')
        self.assertEqual(counts, dict(converted=1, missing=1, ambiguous=0, error=0))
        self.assertTrue(report.exists())
        duplicate = folder.parent / self.source.name
        duplicate.write_bytes((folder / self.source.name).read_bytes())
        counts, _ = batch_convert(self.root / 'input', self.labels, self.root / 'ambiguous')
        self.assertEqual(counts['ambiguous'], 1)
        self.assertEqual(counts['converted'], 0)

    def test_batch_explicit_bkg_filter(self):
        folder = self.root / 'input'
        folder.mkdir()
        self.source.rename(folder / self.source.name.replace('_q20_', '_bkg_'))
        counts, _ = batch_convert(folder, self.labels, self.root / 'bkg', 'bkg')
        self.assertEqual(counts['converted'], 1)


if __name__ == '__main__':
    unittest.main()
