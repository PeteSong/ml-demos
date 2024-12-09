import numpy
import scipy

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

mean_value = numpy.mean(speed)
median_value = numpy.median(speed)
mode_value = scipy.stats.mode(speed)

print(f'mean value: {mean_value},\nmedian value: {median_value},\nmode value: {mode_value}\n')

standard_deviation_value = numpy.std(speed)
print(f'standard deviation: {standard_deviation_value} against the mean value {mean_value}')

variance_value = numpy.var(speed)
print(f'variance: {variance_value}\n')

ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39, 80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31]
q_th = 75
percentile_value = numpy.percentile(ages, 75)
print(f'''
The percentile value: {percentile_value}.
In the ages, there are {q_th}% of the people are {percentile_value} or younger.

''')
