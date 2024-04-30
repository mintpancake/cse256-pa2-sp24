# Alibi Positional Encoding
# Iteration: 99, Train Perplexity: 579.6377563476562, Test Perplexity hbush: 709.1610107421875, Test Perplexity obama: 680.533935546875, Test Perplexity wbush: 777.2838134765625
# Iteration: 199, Train Perplexity: 393.8780517578125, Test Perplexity hbush: 548.2493896484375, Test Perplexity obama: 521.4069213867188, Test Perplexity wbush: 614.6256103515625
# Iteration: 299, Train Perplexity: 268.3271484375, Test Perplexity hbush: 450.506103515625, Test Perplexity obama: 416.6581726074219, Test Perplexity wbush: 512.3653564453125
# Iteration: 399, Train Perplexity: 195.18675231933594, Test Perplexity hbush: 403.4720764160156, Test Perplexity obama: 369.88470458984375, Test Perplexity wbush: 471.4269104003906
# Iteration: 499, Train Perplexity: 152.8769073486328, Test Perplexity hbush: 390.1326904296875, Test Perplexity obama: 349.01202392578125, Test Perplexity wbush: 467.079345703125

# Absolute Positional Encoding
# Iteration: 99, Train Perplexity: 580.19140625, Test Perplexity hbush: 709.6650390625, Test Perplexity obama: 679.6985473632812, Test Perplexity wbush: 772.5736694335938
# Iteration: 199, Train Perplexity: 392.3570556640625, Test Perplexity hbush: 547.9898681640625, Test Perplexity obama: 522.1256713867188, Test Perplexity wbush: 612.529052734375
# Iteration: 299, Train Perplexity: 272.218505859375, Test Perplexity hbush: 463.7820739746094, Test Perplexity obama: 424.9862365722656, Test Perplexity wbush: 522.8549194335938
# Iteration: 399, Train Perplexity: 197.73545837402344, Test Perplexity hbush: 414.879638671875, Test Perplexity obama: 379.97589111328125, Test Perplexity wbush: 480.94805908203125
# Iteration: 499, Train Perplexity: 154.59033203125, Test Perplexity hbush: 394.35595703125, Test Perplexity obama: 352.42041015625, Test Perplexity wbush: 468.0076904296875

import matplotlib.pyplot as plt

# Alibi Positional Encoding
alibi_train_perplexity = [579.6377563476562, 393.8780517578125, 268.3271484375, 195.18675231933594, 152.8769073486328]
alibi_test_perplexity_hbush = [709.1610107421875, 548.2493896484375, 450.506103515625, 403.4720764160156, 390.1326904296875]
alibi_test_perplexity_obama = [680.533935546875, 521.4069213867188, 416.6581726074219, 369.88470458984375, 349.01202392578125]
alibi_test_perplexity_wbush = [777.2838134765625, 614.6256103515625, 512.3653564453125, 471.4269104003906, 467.079345703125]

# Absolute Positional Encoding
absolute_train_perplexity = [580.19140625, 392.3570556640625, 272.218505859375, 197.73545837402344, 154.59033203125]
absolute_test_perplexity_hbush = [709.6650390625, 547.9898681640625, 463.7820739746094, 414.879638671875, 394.35595703125]
absolute_test_perplexity_obama = [679.6985473632812, 522.1256713867188, 424.9862365722656, 379.97589111328125, 352.42041015625]
absolute_test_perplexity_wbush = [772.5736694335938, 612.529052734375, 522.8549194335938, 480.94805908203125, 468.0076904296875]

# Plotting
iterations = [99, 199, 299, 399, 499]

plt.plot(iterations, alibi_test_perplexity_obama, label='Alibi - obama')
plt.plot(iterations, alibi_test_perplexity_wbush, label='Alibi - wbush')
plt.plot(iterations, alibi_test_perplexity_hbush, label='Alibi - hbush')

plt.plot(iterations, absolute_test_perplexity_obama, label='Absolute - obama')
plt.plot(iterations, absolute_test_perplexity_wbush, label='Absolute - wbush')
plt.plot(iterations, absolute_test_perplexity_hbush, label='Absolute - hbush')

plt.xlabel('Iteration')
plt.ylabel('Test Perplexity')
plt.title('Test Perplexity of Alibi and Absolute Positional Encoding')
plt.legend()
plt.savefig('plots\plot_4.png')