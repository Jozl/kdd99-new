import os
import sys

from Code.Svm import svm_kdd99_sp, svm_kdd99_binary_classify, svm_kdd99_use_fake_only, svm_kdd99_use_true_as_test
from Code.Utils.stdout_catch import Logger

file_name = str(os.path.basename(__file__)).replace('py', 'txt')
sys.stdout = Logger(file_name)

classes = ('normal', 'warezclient')
class_lens = (1000, 100)

print('二分类')
svm_kdd99_binary_classify.main(classes, class_lens)
print('kflod对真实数据也均分')
svm_kdd99_sp.main(classes, class_lens)
print('只用fake数据参与分类')
svm_kdd99_use_fake_only.main(classes, class_lens)
print('只用真实数据参与分类')
svm_kdd99_use_true_as_test.main(classes, class_lens)
