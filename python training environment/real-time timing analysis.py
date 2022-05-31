import numpy as np
import sklearn as sk
from sklearn import linear_model
data_array = \
    [[20,13.8628,12.7406],
    [30,20.0215,14.8826],
    [40,28.2156,17.2987],
    [50,39.0493,19.535],
    [60,52.2166,21.5617],
    [70,67.3892,24.1108],
    [80,86.0999,26.6121],
    [90,106.219,28.5315],
    [100,130.470,31.134],
    [110,154.509,33.3099],
    [120,183.456,35.6433]]

data_array = np.array(data_array)

vector_size = data_array[:,0].reshape(-1, 1)
matrix_mult = data_array[:,1].reshape(-1, 1)
activation = data_array[:,2].reshape(-1, 1)

#%%
activation_model = sk.linear_model.LinearRegression()
activation_model.fit(vector_size, activation)
activation_coef = activation_model.coef_[0,0] # 0.2297927272727273
activation_intercept = activation_model.intercept_[0] # 8.038163636363635
#%%
mult_model = sk.linear_model.LinearRegression()
mult_model.fit(vector_size**2, matrix_mult)
mult_coef = mult_model.coef_[0,0] # 0.012093208894638234
mult_intercept = mult_model.intercept_[0] # 8.787240248907139
#%% find critical timing for models
# unit shape is a list of the the units for each layer
# the only thing that isn't accounted for is the vector additions
def model_timing(unit_shape):
    timing = 0
    input_dim = 1
    for units in unit_shape:
        # kernel
        timing += 4*(units*input_dim*mult_coef + mult_intercept)
        # recurrent kernel
        timing += 4*(units**2*mult_coef + mult_intercept)
        # activation. assume tanh costs as much as sigmoid
        timing += 5*(units*activation_coef + activation_intercept)
        # elementwise multiply cost
        timing += 3*(units*mult_coef + mult_intercept)
        input_dim = units
    return timing

cell_sizes = range(1,4)
units_sizes = [40, 35, 30, 25, 20, 18, 15, 12, 10, 8]

timing_results = np.zeros((3, 10))
for i in range(3):
    cells = cell_sizes[i]
    for j in range(10):
        units = units_sizes[j]
        unit_shape = [units for k in range(cells)]
        timing_results[i,j] = model_timing(unit_shape)
