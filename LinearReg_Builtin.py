from sklearn.linear_model import LinearRegression

# Load the data
X = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
X_test = [[0, 1], [10, 2], [20, 5], [30, 9], [40, 12]]
y_test = [4, 7, 15, 22, 25]

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict the labels for the test data
predictions = model.predict(X_test)

# Print the coefficients and intercept of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)
print("Mean squared error:", mse)
