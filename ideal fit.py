import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit

SMALL_SIZE = 12
MEDIUM_SIZE = 15

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend font size

plt.title('Ideal Fit', size=18)
plt.xlabel('Relative Scattering Angle (degrees)', size=15)
plt.ylabel('Coincidence Counts', size=15)

x = np.asarray([90, 75, 60, 45, 30, 15, 0])
y = np.asarray([17221, 17264, 16246, 15858, 14758, 14093, 13740])
xt = np.linspace(90,0,1000)

def sin2law(x,a,b):
    fx = 13740+a*(np.sin(x))**b
    return fx
fa = 54960*(1.25-(np.cos(xt/(18.5*np.pi)))**2)

parameters, covariance = curve_fit(sin2law,x,y)
fit_a = parameters[0]
fit_b = parameters[1]
print('Fitting Parameter')
print(fit_a)
print('Exponent')
print(fit_b)

sin2fit = scipy.optimize.curve_fit(sin2law,x,y)
fit_ydata = sin2law(x,fit_a,fit_b)

plt.plot(xt,fa,label = 'A = 54960', color='white')
plt.plot(xt,fa,label = 'A(1.25-$\cos^2$($\phi$))', color='green')
plt.legend(loc=(.045,.81))

plt.minorticks_on()
plt.tick_params(axis='both', which='major', direction='in', width=2, length=10)
plt.tick_params(axis='both', which='minor', direction='in', width=2, length=5)

plt.yticks(size=12)  # Set label locations.
plt.xticks(np.arange(0, 105, step=15),size=12)  # Set label locations.

plt.grid(True)

plt.savefig('ideal fit.png', dpi=1200)
plt.show()