"""Combine monthly orbit annotation CSVs without losing source columns or rows."""
import argparse
import calendar
import csv
import io
from pathlib import Path
import re


def combine(source, output, workbook=None):
    months = {name.lower(): i for i in range(1, 13)
              for name in (calendar.month_abbr[i], calendar.month_name[i])}
    sources = []
    if workbook:
        import openpyxl
        book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
        for sheet in book:
            match = re.fullmatch(r'AweSolarPanel([A-Za-z]+)(\d{2}|\d{4})', sheet.title, re.I)
            if not match or match[1].lower() not in months:
                continue
            year = int(match[2])
            year += 2000 if year < 100 else 0
            stream = io.StringIO(newline='')
            csv.writer(stream).writerows(sheet.values)
            sources.append((year, months[match[1].lower()], sheet.title, stream.getvalue()))
        book.close()
    else:
        for path in Path(source).glob('*.csv'):
            match = re.fullmatch(r'([A-Za-z]+)(\d{4})', path.stem)
            if match and match[1].lower() in months:
                sources.append((int(match[2]), months[match[1].lower()], path.name,
                                path.read_text(encoding='utf-8-sig')))
    if not sources:
        raise ValueError('No monthly CSV files found')
    fields = ['Year', 'Month', 'Source File', 'Source Row']
    records = []
    for year, month, source_name, content in sorted(sources):
        with io.StringIO(content, newline='') as stream:
            reader = csv.reader(stream)
            headers = next(reader)
            names = []
            for index, header in enumerate(headers, 1):
                name = header.strip() or f'Unnamed column {index}'
                if name == 'Orbit':
                    name = 'Orbit #'
                if name in names:
                    name = f'{name} (column {index})'
                names.append(name)
            for number, values in enumerate(reader, 2):
                row = {'Year': year, 'Month': month, 'Source File': Path(workbook).name if workbook else source_name,
                       'Source Row': number}
                if workbook:
                    row['Source Sheet'] = source_name
                for index, value in enumerate(values):
                    name = names[index] if index < len(names) else f'Unnamed column {index+1}'
                    row[name] = value
                if (year, month) == (2026, 5) and row.get('Orbit #') == '13882':
                    if (row.get('Glare Start 2'), row.get('Glare End 2')) == ('1264', '1258'):
                        row['Glare End 2'] = '1358'
                        row['Correction'] = 'User correction: second glare interval 1264–1358 (2026-10-26)'
                records.append(row)
            for name in names:
                if name not in fields:
                    fields.append(name)
    for row in records:
        for name in row:
            if name not in fields:
                fields.append(name)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f'Saved {len(records)} rows from {len(sources)} monthly sources to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=None)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent/'csv'/'all_months.csv')
    parser.add_argument('--workbook', type=Path, help='Use monthly sheets from this workbook instead of CSVs (requires openpyxl)')
    args = parser.parse_args()
    if args.source is None and args.workbook is None:
        args.workbook = Path(r'C:\Users\domin\Downloads\AweSolarPanels (2).xlsx')
    if args.source is not None and args.workbook is not None:
        parser.error('Choose --source or --workbook, not both')
    combine(args.source, args.output, args.workbook)
