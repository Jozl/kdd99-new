import csv


class SheetWriter:
    def __init__(self, file_name='sheet_saved.csv'):
        self.csv_writer = csv.writer(open(file_name, 'w', newline=''))

    def writerow(self, row):
        self.csv_writer.writerow(row)


if __name__ == '__main__':
    c = SheetWriter()
    c.writerow(['acc+', 'acc-', 'accuracy', 'precision', 'recall', 'F1', 'G-mean'])
