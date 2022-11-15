import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
data_array = np.loadtxt("timing results.csv", delimiter=',')

data_array = np.array(data_array)

vector_size = np.array(range(20, 130, 10)).reshape(-1, 1)
matrix_mult = data_array[:,0].reshape(-1, 1)
activation = data_array[:,1].reshape(-1, 1)
vector_mult = data_array[:,2].reshape(-1, 1)
vector_add = data_array[:,3].reshape(-1, 1)

#%% activation 
activation_model = sk.linear_model.LinearRegression()
activation_model.fit(vector_size, activation)
s1 = activation_model.coef_[0,0] # 0.2297927272727273
s0 = activation_model.intercept_[0] # 8.038163636363635
#%% matrix multiplication
# poly = PolynomialFeatures(degree=2, include_bias=False)
# poly_features = poly.fit_transform(vector_size)
# mult_model = sk.linear_model.LinearRegression()
# mult_model.fit(poly_features, matrix_mult)
# mult_coef = mult_model.coef_[0,0] # 0.012093208894638234
# mult_intercept = mult_model.intercept_[0] # 8.787240248907139
mult_model = sk.linear_model.LinearRegression()
mult_model.fit(vector_size**2, matrix_mult)
m1 = mult_model.coef_[0,0] # 0.012093208894638234
m0 = mult_model.intercept_[0] # 8.787240248907139
#%% vector add
vadd_model = sk.linear_model.LinearRegression()
vadd_model.fit(vector_size, vector_add)
a1 = vadd_model.coef_[0,0]
a0 = vadd_model.intercept_[0]
#%% vector multiply
vmult_model = sk.linear_model.LinearRegression()
vmult_model.fit(vector_size, vector_mult)
b1 = vmult_model.coef_[0,0]
b0 = vmult_model.intercept_[0]
#%%
print("s1: " + str(s1))
print("s0: " + str(s0))
print("m1: " + str(m1))
print("m0: " + str(m0))
print("a1: " + str(a1))
print("a0: " + str(a0))
print("b1: " + str(b1))
print("b0: " + str(b0))

c1 = 4*m1
c2 = 5*s1 + 9*a1 + 3*b1
c3 = 5*s0 + 8*m0 + 9*a0 + 3*b0

time_layer = lambda u, n: c1*u*(u+n) + c2*u + c3
def time_model(input_shape, units):
    i = input_shape
    t = 0
    for u in units:
        t += time_layer(u, i)
        i = u
    return t

units = [30, 30]
t = time_model(1, units)
print(t)
# make critical timing chart
units = [40, 35, 30, 25, 20, 18, 15, 12, 10, 8]
critical_times = np.zeros((10, 3))
for i, num_units in enumerate(range(1, 4)):
    for j, u in enumerate(units):
        u_shape = [u]*num_units
        t = time_model(16, u_shape)
        critical_times[j, i] = t
#%% find critical timing for models
# unit shape is a list of the the units for each layer
# the only thing that isn't accounted for is the vector additions
# def model_timing(unit_shape):
#     timing = 0
#     input_dim = 1
#     for units in unit_shape:
#         # kernel
#         timing += 4*(units*input_dim*mult_coef + mult_intercept)
#         # recurrent kernel
#         timing += 4*(units**2*mult_coef + mult_intercept)
#         # activation. assume tanh costs as much as sigmoid
#         timing += 5*(units*activation_coef + activation_intercept)
#         # elementwise multiply cost
#         timing += 3*(units*mult_coef + mult_intercept)
#         input_dim = units
#     return timing

# cell_sizes = range(1,4)
# units_sizes = [40, 35, 30, 25, 20, 18, 15, 12, 10, 8]

# timing_results = np.zeros((3, 10))
# for i in range(3):
#     cells = cell_sizes[i]
#     for j in range(10):
#         units = units_sizes[j]
#         unit_shape = [units for k in range(cells)]
#         timing_results[i,j] = model_timing(unit_shape)
