import os


class DataLoader:
    @staticmethod
    def load(data_name):
        root_path = os.path.join(os.path.abspath('../../'), 'Data')
        return open(DataLoader.search_file(root_path, data_name))

    @staticmethod
    def search_file(root, target):
        os.chdir(root)
        items = os.listdir(os.curdir)
        for item in items:
            path = os.path.join(root, item)
            if os.path.isdir(path):
                res = DataLoader.search_file(path, target)
                if res:
                    return res
                os.chdir(os.pardir)
            elif path.split(os.sep)[-1] == target:
                return path
        return None


if __name__ == '__main__':
    f = DataLoader.load('ecoli4.dat')
    print(f.readline())
