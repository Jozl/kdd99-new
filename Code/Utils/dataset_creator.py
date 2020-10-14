import os

path_root = os.path.abspath('../../')
file_head = os.path.join(path_root, 'Data', 'kdd99_head.dat')


def dataset_create(classes, classes_len):
    file_output = os.path.join(path_root, 'Data', '多分类 data3', 'kdd99_new_multi.dat')
    outputs = open(file_output, 'w')
    with open(file_head) as f:
        for row in f:
            outputs.write(row)
        outputs.write('\n')

    for clazz, clazz_len in zip(classes, classes_len):
        inputs = open(os.path.join(path_root, 'Data', 'KDD99', 'kdd99_{}.kdd99'.format(clazz)))
        for i, row in enumerate(inputs):
            if i == clazz_len:
                break
            outputs.write(row)

        inputs.close()
    outputs.close()

    return file_output.split(os.sep)[-1]


if __name__ == '__main__':
    data_name = dataset_create(['normal', 'warezclient'], (1000, 60))
    print(data_name)
