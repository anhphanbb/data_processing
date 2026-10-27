import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('glare', ROOT/'0_copy_and_remove_glare.py')
glare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(glare)


class CombinedGlareTests(unittest.TestCase):
    def test_month_isolation_and_two_intervals(self):
        data = pd.DataFrame([
            {'Year': 2023, 'Month': 11, 'Orbit #': 2, 'Glare': 1,
             'Glare Start 1': 0, 'Glare End 1': 1, 'Glare Start 2': 4, 'Glare End 2': 4},
            {'Year': 2023, 'Month': 12, 'Orbit #': 2, 'Glare': 1,
             'Glare Start 1': 7, 'Glare End 1': 8},
        ])
        intervals = glare.extract_glare_intervals(data, 2023, 11, 2)
        self.assertEqual(intervals, {2: [(0, 1), (4, 4)]})
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            source = Path(tmp)/'awe_l0c_q20_2023326T0236_00002_v02.nc'
            target = Path(tmp)/'result.nc'
            with Dataset(source, 'w') as ds:
                ds.createDimension('time', 30)
                ds.createVariable('Epoch', 'i4', ('time',))[:] = np.arange(30)
            glare.remove_glare_and_save(str(source), intervals, str(target))
            with Dataset(target) as ds:
                self.assertEqual(ds['Epoch'][:].tolist(), [12, 13, 15, 16, 17, 18, 19])
        with self.assertRaisesRegex(ValueError, 'no coverage'):
            glare.extract_glare_intervals(data, 2024, 1, 2)

    def test_legacy_equivalence(self):
        fixtures = [ROOT/f'csv/IndividualMonths/glare_intervals_{name}.csv'
                    for name in ('Nov2023', 'Dec2023')]
        if not all(path.is_file() for path in fixtures):
            self.skipTest('Optional legacy monthly CSV fixtures are not included in this repository')
        combined = pd.read_csv(ROOT/'csv/all_months.csv', low_memory=False)
        for month, name in ((11, 'Nov2023'), (12, 'Dec2023')):
            legacy = pd.read_csv(ROOT/f'csv/IndividualMonths/glare_intervals_{name}.csv')
            self.assertEqual(glare.extract_glare_intervals(legacy),
                             glare.extract_glare_intervals(combined, 2023, month))

    def test_workbook_replaces_bad_extraction_and_corrects_interval(self):
        data = pd.read_csv(ROOT/'csv/all_months.csv', low_memory=False)
        feb = data[(data.Year == 2025) & (data.Month == 2)]
        self.assertEqual(int(feb['Orbit #'].dropna().iloc[0]), 6772)
        self.assertNotIn(13882, feb['Orbit #'].dropna().tolist())
        self.assertEqual(glare.extract_glare_intervals(data, 2026, 5, 13882),
                         {13882: [(0, 295), (1264, 1358)]})
        for (year, month), rows in data.groupby(['Year', 'Month']):
            glare.extract_glare_intervals(rows, int(year), int(month))


if __name__ == '__main__':
    unittest.main()
