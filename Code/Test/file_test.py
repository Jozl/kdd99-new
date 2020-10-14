with open('file_test.py') as f:
    for row, i in zip(f, range(2)):
        print(row)
    for row in f:
        print(row)
