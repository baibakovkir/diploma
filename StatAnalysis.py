import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

df = pd.read_excel('protein_Belgorod.xlsx')

max_temp_list = df['t max'].dropna().tolist()
humidity_list = df['r humidity'].dropna().tolist()
protein_list = df['protein'].dropna().tolist()
year_list = df['year'].dropna().tolist()
total_years = df['total years'].dropna().tolist()
total_max = df['total max'].dropna().tolist()
total_humidity = df['total humidity'].dropna().tolist()
protein_full = df['protein_full'].dropna().tolist()
T_v = df['T_v'].dropna().tolist()
R_v = df['R_v'].dropna().tolist()


# # Create a matrix of independent variables (X) including 't max' and 'r humidity'
# X = np.column_stack((max_temp_list, humidity_list))

# Z = np.column_stack((total_max, total_humidity))


# # Create the dependent variable (y) using the original protein values
# y = np.array(protein_list)

# # Create a polynomial regression model
# degree = 2  # You can adjust the degree of the polynomial
# model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
# model.fit(X, y)


# # Use the model to predict protein values for the elements that have a value of 0
# recreated_protein_list = model.predict(Z)

# print("recreated_protein_list", recreated_protein_list)


# # Create a meshgrid of 't max' and 'r humidity' values for visualization
# t_max_values = np.linspace(min(max_temp_list), max(max_temp_list), 100)
# humidity_values = np.linspace(min(humidity_list), max(humidity_list), 100)
# T_max, Humidity = np.meshgrid(t_max_values, humidity_values)
# Z_mesh = np.column_stack((T_max.ravel(), Humidity.ravel()))
# predicted_proteins = model.predict(Z_mesh).reshape(T_max.shape)

# print(predicted_proteins)

# # Create a 3D plot to visualize the regression surface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(max_temp_list, humidity_list, protein_list, color='blue', label='Actual Protein Values')
# ax.plot_surface(T_max, Humidity, predicted_proteins, alpha=0.5, color='green', label='Regression Surface')

# ax.set_xlabel('Max Temperature')
# ax.set_ylabel('Humidity')
# ax.set_zlabel('Protein Content')
# ax.set_title('Polynomial Regression 3D Plot')
# plt.legend()
# plt.show()
print(T_v)
print(R_v)
print(protein_full)
# Create a matrix of independent variables (X) including 't max' and 'r humidity'
X = np.column_stack((T_v, R_v))
X = sm.add_constant(X)  # Add a constant term for the intercept

# Create the dependent variable (y) using the original protein values
y = np.array(protein_full)

# Fit the OLS model using stepwise regression
model = sm.OLS(y, X).fit()

# Display the summary table of the stepwise regression results
print(model.summary())
