# Alibi Positional Encoding
# Epoch: 0, Train Accuracy: 46.41491395793499, Test Accuracy: 37.06666666666667
# Epoch: 1, Train Accuracy: 52.48565965583174, Test Accuracy: 44.93333333333333
# Epoch: 2, Train Accuracy: 56.978967495219884, Test Accuracy: 50.13333333333333
# Epoch: 3, Train Accuracy: 66.87380497131932, Test Accuracy: 56.266666666666666
# Epoch: 4, Train Accuracy: 76.33843212237093, Test Accuracy: 65.33333333333333
# Epoch: 5, Train Accuracy: 81.83556405353728, Test Accuracy: 71.06666666666666
# Epoch: 6, Train Accuracy: 87.8585086042065, Test Accuracy: 75.46666666666667
# Epoch: 7, Train Accuracy: 91.49139579349904, Test Accuracy: 79.2
# Epoch: 8, Train Accuracy: 94.93307839388146, Test Accuracy: 83.86666666666666
# Epoch: 9, Train Accuracy: 95.26768642447419, Test Accuracy: 83.2
# Epoch: 10, Train Accuracy: 93.49904397705545, Test Accuracy: 81.46666666666667
# Epoch: 11, Train Accuracy: 97.56214149139579, Test Accuracy: 84.93333333333334
# Epoch: 12, Train Accuracy: 97.51434034416826, Test Accuracy: 85.33333333333333
# Epoch: 13, Train Accuracy: 98.61376673040154, Test Accuracy: 85.33333333333333
# Epoch: 14, Train Accuracy: 99.1395793499044, Test Accuracy: 86.26666666666667

# Absolute Positional Encoding
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



# Alibi Positional Encoding
alibi_test_accuracy = [37.06666666666667, 44.93333333333333, 50.13333333333333, 56.266666666666666, 65.33333333333333, 71.06666666666666, 75.46666666666667, 79.2, 83.86666666666666, 83.2, 81.46666666666667, 84.93333333333334, 85.33333333333333, 85.33333333333333, 86.26666666666667]

# Absolute Positional Encoding
absolute_test_accuracy = [33.333333333333336, 33.333333333333336, 49.6, 57.06666666666667, 64.8, 70.8, 72.8, 79.33333333333333, 82.4, 83.86666666666666, 81.06666666666666, 84.26666666666667, 83.06666666666666, 85.33333333333333, 85.73333333333333]

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

plt.plot(epochs, alibi_test_accuracy, label='Alibi Positional Encoding')
plt.plot(epochs, absolute_test_accuracy, label='Absolute Positional Encoding')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison')
plt.legend()
plt.savefig('plots\plot_3.png')