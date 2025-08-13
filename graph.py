import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
epochs = np.arange(1, 101)  # 1 to 100 epochs
loss = np.exp(-0.05 * epochs) + np.random.normal(0, 0.02, size=epochs.shape)  # Simulated loss values

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='Loss', color='blue')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.grid(True)
plt.legend()
plt.show()  # Display the plot
