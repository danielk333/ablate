import scipy
import numpy as np
import matplotlib.pyplot as plt

from ablate.functions.ablation import alpha_beta_Q4_min as Q4_min
from ablate.functions.ablation import alpha_beta_Q4_min as min_fun

f = '/home/danielk/data/test_data/DN150417.csv'
slope = 15.17

v0 = []

vel_col = "D_DT_geo"
h_col = "height"

data = np.genfromtxt(f, delimiter=',', dtype="f8,f8,f8", names=["height","D_DT_geo","D_DT_fitted"])
slope = np.deg2rad(slope)

alt = []#np.asarray(data['height'])
vel = []#np.asarray(data['D_DT_geo'])

# remove any nan values
for v in range(len(data[vel_col])):
    if data[vel_col][v] >1.:
        vel.append(data[vel_col][v])
        alt.append(data[h_col][v])
        
# define initial velocity, if not already
if v0 == []:
    v0 = np.nanmean(vel[0:10])

# normalise velocity
vel = np.asarray(vel)
alt = np.asarray(alt)
Vvalues = vel/v0      #creates a matrix of V/Ve to give a dimensionless parameter for velocity


# normalise height - if statement accounts for km vs. metres data values.
if alt[0]<1000:
    h0 = 7.160  # km
else:
    h0 = 7160.  # metres
Yvalues = alt/h0  

Gparams= Q4_min(Vvalues, Yvalues)

alpha = Gparams[0]
beta = Gparams[1]

print(alpha, beta)

plt.close()
# plt.rcParams['figure.dpi'] = 10
plt.rcParams['figure.figsize'] = [5, 5]

x = np.arange(0.1,1, 0.00005);                                                                                     #create a matrix of x values
fun = lambda x:np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta* x**2) )/2)
y = [fun(i) for i in x]

plt.scatter(Vvalues, Yvalues,marker='x', label=None)  
plt.xlabel("normalised velocity")
plt.ylabel("normalised height")
plt.plot(x, y, color='r')
# plt.xlim(0.4, 1.23)
# plt.ylim(6, 12)
plt.show()

plt.close()
plt.rcParams['figure.figsize'] = [7, 7]

# define x values
x_mu = np.arange(0,10, 0.00005)

# function for mu = 0, 50 g possible meteorite:
fun_mu0 = lambda x_mu:np.log(13.2 - 3*x_mu)
y_mu0 = [fun_mu0(i) for i in x_mu]

# function for mu = 2/3, 50 g possible meteorite:
fun_mu23 = lambda x_mu:np.log(4.4 - x_mu)
y_mu23 = [fun_mu23(i) for i in x_mu]

# plot mu0, mu2/3 lines and your poit:
plt.plot(x_mu, y_mu0, color='grey')
plt.plot(x_mu, y_mu23, color='k')
plt.scatter([np.log(alpha * np.sin(slope))], [np.log(beta)], color='r')

# defite plot parameters
plt.xlim((-1, 7))
plt.ylim((-3, 4))
plt.xlabel("ln(alpha x sin(slope))")
plt.ylabel("ln(beta)")
plt.axes().set_aspect('equal')
plt.show()

