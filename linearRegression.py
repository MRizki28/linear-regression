import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FuncFormatter

# Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([100000000, 200000000, 300000000, 400000000, 500000000])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Model
model = LinearRegression()

# Fitting model
model.fit(X_train, y_train)

# Prediction
y_pred_train = model.predict(X_train)  # Predictions on training data
y_pred_test = model.predict(X_test)    # Predictions on test data

# Plot data and predictions
plt.scatter(X_train, y_train, color='blue', label='Data Latih')
plt.scatter(X_test, y_test, color='red', label='Data Uji')
plt.plot(X, model.predict(X), color='green', label='Prediksi')  # Use entire X range for plotting predictions
plt.xlabel('Jumlah Kamar Tidur')
plt.ylabel('Harga Rumah')
plt.title('Prediksi Harga Rumah dengan Regresi Linier')
plt.legend()

# Formatter for y axis
def millions(x, pos):
    return '{:,.0f}'.format(x)

formatter = FuncFormatter(lambda x, pos: 'Rp {:,.0f}'.format(x))
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()
