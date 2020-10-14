import matplotlib.pyplot as plt

a = [1,2,2,3,3,3,4,4,4,4,5,5,5,6,6,7]

plt.hist(a, bins=100, facecolor='green', density=True)

plt.title('data name {} atr : {}')
plt.xlabel(' temperature')
plt.ylabel('Probability')
# 输出
plt.show()
