from sklearn.linear_model import LinearRegression

# Define the data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([0, 0, 1, 1])

# Create the model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Print the coefficients
print(model.coef_)
