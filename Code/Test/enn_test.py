from Code.Utils.dataset_creator import dataset_create
from Code.Svm.svm import do_classify

classes = ('normal', 'teardrop', 'nmap', 'ipsweep', 'buffer_overflow', 'warezclient')
classes = ('normal', 'teardrop', 'ipsweep')

classes = ('normal', 'warezclient')
class_len = (1000, 345)

data_name = dataset_create(classes, class_len)
print(data_name)
do_classify(data_name, binary_class=True, using_kdd99=True)
