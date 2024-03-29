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
total_years = df['total years 2'].dropna().tolist()
total_max = df['total max'].dropna().tolist()
total_humidity = df['total humidity'].dropna().tolist()
protein_full = df['protein_full'].dropna().tolist()
T_v = df['T_v'].dropna().tolist()
R_v = df['R_v'].dropna().tolist()
E_v = df['E_v'].dropna().tolist()
Pss_v = df['Pss_v'].dropna().tolist()
Ob_v = df['Ob_v'].dropna().tolist()
HTC_v = df['HTC_v'].dropna().tolist()
T_v_full = df['T_v_full'].dropna().tolist()
R_v_full = df['R_v_full'].dropna().tolist()
E_v_full = df['E_v_full'].dropna().tolist()
Pss_v_full = df['Pss_v_full'].dropna().tolist()
Ob_v_full = df['Ob_v_full'].dropna().tolist()
HTC_v_full = df['HTC_v_full'].dropna().tolist()

# Create a matrix of independent variables (X) including 't max' and 'r humidity'
X = np.column_stack((T_v, R_v, E_v, Pss_v, Ob_v, HTC_v))
X = sm.add_constant(X)  # Add a constant term for the intercept

# Create the dependent variable (y) using the original protein values
y = np.array(protein_full)

# Fit the OLS model using stepwise regression
model = sm.OLS(y, X).fit(stepwise=True)

# Display the summary table of the stepwise regression results
print(model.summary())

# Create a matrix of independent variables (Z) including 't max' and 'r humidity'
Z = np.column_stack((T_v_full, R_v_full, E_v_full, Pss_v_full, Ob_v_full, HTC_v_full))
Z = sm.add_constant(Z)  # Add a constant term for the intercept

# Create the dependent variable (y) using the original protein values
y = np.array(protein_full)

# Create a polynomial regression model
degree = 1  # You can adjust the degree of the polynomial
model2 = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model2.fit(X, y)

# Use the model to predict protein values for the elements that have a value of 0
recreated_protein_list = model.predict(Z)

protein_forecast = model.predict(X)

print("recreated_protein_list", recreated_protein_list)

plt.scatter(total_years, model.predict(X), color='blue', label='Прогноз')
plt.plot(total_years, model.predict(X), color='blue', linestyle='-', linewidth=1)  # Add line connecting the points
plt.scatter(total_years, y, color='red', label='Факт')
plt.plot(total_years, y, color='red', linestyle='-', linewidth=1)  # Add line connecting the points
plt.xlabel('Год')
plt.ylabel('Значение белка')
plt.title('Прогнозируемое и фактическое значение белка')
plt.legend()
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = protein_full
# y = model2.predict(X)
# X, Y = np.meshgrid(x, y)
# Z = model.predict(X)

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')

# ax.set_xlabel('Fact Protein Values')
# ax.set_ylabel('Predicted Protein Values')
# ax.set_zlabel('Predicted Protein Values')

# plt.show()

errors = [abs(pred - actual) for pred, actual in zip(protein_forecast, y)]
avg_error = sum(errors) / len(errors)

print("Average error of the forecast:", avg_error)

formula = model.params

print("Formula:")
print(formula)