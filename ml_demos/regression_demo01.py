import matplotlib.pyplot as plt
import numpy
import scipy
from sklearn.metrics import r2_score

x1 = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y1 = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
# x1 = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
# y1 = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
slope, intercept, r, p, std_err = scipy.stats.linregress(x1, y1)
print(f'linear regression => slope:{slope}, intercept:{intercept}, r-value: {r}, p-value: {p}, err: {std_err}\n')


def linear_func(x):
    return slope * x + intercept


model1 = list(map(linear_func, x1))

######

x2 = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y2 = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
model2 = numpy.poly1d(numpy.polyfit(x2, y2, 3))
r2_value = r2_score(y2, model2(x2))
print(f'polynomial regression => r2-score: {r2_value}\n')
line2 = numpy.linspace(1, 22, 100)

######

fig, axs = plt.subplots(1, 2)

axs[0].set_title('Linear regression')
axs[0].scatter(x1, y1)
axs[0].plot(x1, model1)

axs[1].set_title('Polynomial regression')
axs[1].scatter(x2, y2)
axs[1].plot(line2, model2(line2))

plt.show()

#######
