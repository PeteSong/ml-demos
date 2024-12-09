import numpy
import matplotlib.pyplot as plt

# draw data from uniform data distribution 均匀分布
x = numpy.random.uniform(0.0, 5.0, 100000)
# plt.hist(x, 100)
# plt.show()

# draw data from normal/Gaussian data distribution 正太/高斯分布
y = numpy.random.normal(2.5, 0.8, 100000)
# plt.hist(y, 100)
# plt.show()

a = numpy.random.normal(5, 1, 1000)
b = numpy.random.normal(10, 2, 1000)
# plt.scatter(x, y)
# plt.show()

fig, axs = plt.subplots(1, 3, sharex=True)
axs[0].set_title('Uniform data distribution')
axs[0].hist(x, 100)

axs[1].set_title('Normal data distribution')
axs[1].hist(y, 100)

axs[2].set_title('Scatter plot')
axs[2].scatter(a, b)
plt.show()