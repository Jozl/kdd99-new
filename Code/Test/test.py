from Code.Dataset.dataset import MyDataSet

d = MyDataSet('kdd99_new_multi.dat')
print(d.datalist[-2])
print(d.get_original_datalist[-2])