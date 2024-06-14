import math
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

import numpy as np

class BSM:
    
    def __init__(self, strike_price, current_price, time, rate, volatility):
        self.K = strike_price
        self.S = current_price
        self.T = time
        self.r = rate
        self.sigma = volatility

    def calculate_d1(self):
        d1 = (math.log(self.S / self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma * math.sqrt(self.T))
        return d1

    def calculate_d2(self):
        d2 = self.calculate_d1() - self.sigma * math.sqrt(self.T)
        return d2

    def N(self, x):
        ans = norm.cdf(x)
        return ans

    def call_price(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        Vc = self.S * self.N(d1) - self.K * (math.exp(-1*self.r*self.T)) * self.N(d2)
        return Vc

    def put_price(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        Vp = self.K * (math.exp(-1*self.r*self.T)) * self.N(-1*d2) - self.S * self.N(-1*d1)
        return Vp

# ----------------------------------------------------------------------------------------
# This is the first part that creates a line plot

tbr = [5.183, 5.368, 5.391, 5.379, 5.183] # tresury bond rate
infl = 3.36 # inflation rate
rfr = [1.703, 1.888, 1.911, 1.899, 1.703] # Risk free rates = tbr - infl

time = [0.02083, 0.083, 0.25, 0.5, 1] # 1 Week, 1 Month, 3 Months, 6 Months, 1 Year
strikes = [4720, 4825, 5030, 5200, 5320, 5340, 5450] # Strikes price array
volatilities = [0.5078, 0.418, 0.2627, 0.1141, 0.0454, 0.0995, 0.2704] # Volatilities


current = 5303.27 # Spot Price

results = []
for i in range(5):
    t = time[i].

    
    r = rfr[i] / 100
    for (k, sigma) in zip(strikes, volatilities):
        bsm = BSM(k, current, t, r, sigma)
        call_price = bsm.call_price()
        put_price = bsm.put_price()
        results.append([k, t, call_price, put_price])

df = pd.DataFrame(results, columns=['Strike', 'Expiry', 'Call Price', 'Put Price'])

print(df)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = df['Strike']
Y = df['Expiry']
Z_call = df['Call Price']
Z_put = df['Put Price']
ax.plot(X, Y, Z_call, c='r', marker='o', label='Call Prices')
ax.plot(X, Y, Z_put, c='b', marker='o', label='Put Prices')

ax.set_xlabel('Strike Prices')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Option Prices')
ax.set_title('Option Prices vs. Strike Prices and Time to Expiry')

ax.legend()

plt.show()

# -----------------------------------------------------------------------------------------
# This is the second part of the code, basically different input values and creating a surf plot
'''
strike_prices = np.linspace(200, 5500, 100)
time = np.linspace(0.0192307692, 1.0, 100)
sigma = 0.5
r  =0.01
current = 5293

results = []

for t in time:
    for k in strike_prices:
        bsm = BSM(k, current, t, r, sigma)
        call_price = bsm.call_price()
        put_price = bsm.put_price()
        results.append([k, t, call_price, put_price])

df = pd.DataFrame(results, columns=['Strike', 'Expiry', 'Call Price', 'Put Price'])

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_trisurf(df['Strike'], df['Expiry'], df['Call Price'], cmap='viridis')
ax1.set_title('Call Price vs Strike vs Time')
ax1.set_xlabel('Strike Price')
ax1.set_ylabel('Time to Expiry')
ax1.set_zlabel('Call Price')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_trisurf(df['Strike'], df['Expiry'], df['Put Price'], cmap='plasma')
ax2.set_title('Put Price vs Strike vs Time')
ax2.set_xlabel('Strike Price')
ax2.set_ylabel('Time to Expiry')
ax2.set_zlabel('Put Price')

plt.tight_layout()
plt.show()

'''
