# Improved
# Epoch: 0, Train Accuracy: 44.64627151051625, Test Accuracy: 33.333333333333336
# Epoch: 1, Train Accuracy: 52.58126195028681, Test Accuracy: 43.333333333333336
# Epoch: 2, Train Accuracy: 63.718929254302104, Test Accuracy: 60.4
# Epoch: 3, Train Accuracy: 70.26768642447419, Test Accuracy: 64.13333333333334
# Epoch: 4, Train Accuracy: 72.65774378585085, Test Accuracy: 65.73333333333333
# Epoch: 5, Train Accuracy: 81.83556405353728, Test Accuracy: 70.53333333333333
# Epoch: 6, Train Accuracy: 85.94646271510516, Test Accuracy: 73.86666666666666
# Epoch: 7, Train Accuracy: 90.43977055449331, Test Accuracy: 79.6
# Epoch: 8, Train Accuracy: 93.59464627151051, Test Accuracy: 83.46666666666667
# Epoch: 9, Train Accuracy: 93.97705544933078, Test Accuracy: 83.73333333333333
# Epoch: 10, Train Accuracy: 95.50669216061185, Test Accuracy: 84.4
# Epoch: 11, Train Accuracy: 96.12810707456978, Test Accuracy: 84.26666666666667
# Epoch: 12, Train Accuracy: 96.65391969407266, Test Accuracy: 85.73333333333333
# Epoch: 13, Train Accuracy: 96.98852772466539, Test Accuracy: 85.73333333333333
# Epoch: 14, Train Accuracy: 97.131931166348, Test Accuracy: 86.0
# Epoch: 15, Train Accuracy: 97.32313575525812, Test Accuracy: 86.13333333333334
# Epoch: 16, Train Accuracy: 97.7055449330784, Test Accuracy: 86.53333333333333
# Epoch: 17, Train Accuracy: 98.04015296367113, Test Accuracy: 86.66666666666667
# Epoch: 18, Train Accuracy: 97.99235181644359, Test Accuracy: 86.93333333333334
# Epoch: 19, Train Accuracy: 98.27915869980879, Test Accuracy: 86.93333333333334

# Previous
# Epoch: 0, Train Accuracy: 44.64627151051625, Test Accuracy: 33.333333333333336
# Epoch: 1, Train Accuracy: 44.64627151051625, Test Accuracy: 33.333333333333336
# Epoch: 2, Train Accuracy: 56.35755258126195, Test Accuracy: 49.6
# Epoch: 3, Train Accuracy: 64.5793499043977, Test Accuracy: 57.06666666666667
# Epoch: 4, Train Accuracy: 74.90439770554494, Test Accuracy: 64.8
# Epoch: 5, Train Accuracy: 80.87954110898661, Test Accuracy: 70.8
# Epoch: 6, Train Accuracy: 84.51242829827916, Test Accuracy: 72.8
# Epoch: 7, Train Accuracy: 91.92160611854685, Test Accuracy: 79.33333333333333
# Epoch: 8, Train Accuracy: 94.07265774378585, Test Accuracy: 82.4
# Epoch: 9, Train Accuracy: 95.88910133843213, Test Accuracy: 83.86666666666666
# Epoch: 10, Train Accuracy: 93.78585086042065, Test Accuracy: 81.06666666666666
# Epoch: 11, Train Accuracy: 97.4187380497132, Test Accuracy: 84.26666666666667
# Epoch: 12, Train Accuracy: 96.17590822179733, Test Accuracy: 83.06666666666666
# Epoch: 13, Train Accuracy: 97.89674952198853, Test Accuracy: 85.33333333333333
# Epoch: 14, Train Accuracy: 98.18355640535373, Test Accuracy: 85.73333333333333

import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Improved
test_accuracy_improved = [33.333333333333336, 43.333333333333336, 60.4, 64.13333333333334, 65.73333333333333, 70.53333333333333, 73.86666666666666, 79.6, 83.46666666666667, 83.73333333333333, 84.4, 84.26666666666667, 85.73333333333333, 85.73333333333333, 86.0, 86.13333333333334, 86.53333333333333, 86.66666666666667, 86.93333333333334, 86.93333333333334]

# Previous
test_accuracy_previous = [33.333333333333336, 33.333333333333336, 49.6, 57.06666666666667, 64.8, 70.8, 72.8, 79.33333333333333, 82.4, 83.86666666666666, 81.06666666666666, 84.26666666666667, 83.06666666666666, 85.33333333333333, 85.73333333333333]

# Plotting
plt.plot(epochs, test_accuracy_improved, label='Improved')
plt.plot(epochs[:15], test_accuracy_previous, label='Previous')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison')
plt.legend()
plt.savefig('plots\plot_5.png')