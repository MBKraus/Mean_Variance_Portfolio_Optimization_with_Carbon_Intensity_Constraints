import cvxopt as opt
from cvxopt import blas, solvers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl

# Source returns from Quandl (NB - Quandl isn't actively supported anymore post April 11, 2018)

tickers = ['AMZN', 'GOOGL', 'TSLA']
start_date='2016-01-01'
end_date='2018-01-01'
returns = pd.DataFrame(dict([(ticker, quandl.get('WIKI/'+ticker,
                                    start_date=start_date,
                                    end_date=end_date)['Adj. Close'].pct_change())
                for ticker in tickers]))
returns = returns.fillna(method='ffill').iloc[1:]
returns = returns * 100
returns = returns.as_matrix().T
#eturns = returns.T

n = len(returns)
mu_r = np.mean(returns, axis=1)

# Optional for saving to CSV and loading it accordingly (along this way you don't need to call Quandl every time you run an optimisation)

data.to_csv("datatot.csv", sep=';', encoding='utf-8')
data = pd.read_csv('datatot.csv', sep=';')
data.set_index('Date', inplace=True)

data.sort_index(inplace=True, ascending=True)
data.index = pd.to_datetime(data.index)
returns = data.pct_change().dropna()

# Set scope 1+2 carbon intensity for each stock (NB - these numbers are fictive)

carbon_dict = {2016: {'AMZN': 10.9, 'GOOGL': 17.1, 'TSLA': 13.8},
                 2017: {'AMZN': 8.7, 'GOOGL': 17.4, 'TSLA': 13.6},
                2018: {'AMZN': 11.8, 'GOOGL': 17.6, 'TSLA': 13.9},
                }

carbon_df = pd.DataFrame.from_dict(carbon_dict, orient="index")
carbon = carbon_df.as_matrix().T

mu_carbon = np.mean(carbon, axis=1)

# Minimize variance with respect to:
# 1) the weight parameters for a given vector of mean daily returns and
# 2) with respect to a target weighted portfolio carbon intensity (of 15)
# This along portfolio constraints that:
# a) stipulate that all portfolio weights sum to one and
# b) only long positions can be taken on in the portfolio.

N = 10
mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

# Convert covariance matrix and mean return vector to cvxopt matrices

S = opt.matrix(np.cov(returns))
pbar = opt.matrix(np.mean(returns, axis=1))

# Create constraint matrices for the following constraints:
# 1) the portfolio can only hold long-only positions
# 2) the portfolio weights have to sum up to 1
# 3) a weighted portfolio scope 1+2 carbon intensity of 15 would have to be met

G = -opt.matrix(np.eye(n))

h = opt.matrix(0.0, (n ,1))

A_ones = np.matrix(np.ones((1,n)))

A = opt.matrix(np.vstack((A_ones, mu_carbon.T)))

b = opt.matrix(np.matrix([1.0, 15]).T)

# Calculate efficient frontier weights using CVXOPT

portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]

# Calculate risks and returns for the efficient frontier

returns = [blas.dot(pbar, x) for x in portfolios]
risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

# Calculate the 2nd degree polynomial of the efficient frontier

m1 = np.polyfit(returns, risks, 2)
x1 = np.sqrt(m1[2] / m1[0])

# Plot efficient frontier

plt.style.use('seaborn')
fig = plt.figure()
plt.ylabel('Mean daily return')
plt.xlabel('Standard deviation')
plt.plot(risks, returns, 'b-o', lw=1, alpha=0.4)
plt.title('Efficient Frontier')
plt.text(1.1310,0.09205, ' Portfolio Scope 1+2 Carbon Intensity = {}'.format(float(np.dot(mu_carbon.T, portfolios[9]))))
plt.text(1.1310,0.092035, ' Return - Global Mean Variance Portfolio = {}'.format(float(np.dot(mu_r.T, portfolios[9]))))
plt.text(1.1310,0.09202, ' STD - Global Mean Variance Portfolio = {}'.format(float(risks[9])))
plt.show()

# Plot the portfolio weights of the global mean variance portfolio in a pie chart

labels = ['Amazon', 'Google', 'Tesla']
patches, texts, autotexts = plt.pie(portfolios[9], labels = labels, startangle=90, autopct='%1.1f%%')
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.title('Global Minimum Variance Portfolio')
plt.show()

# Print portfolio weights, return, standard deviation and weighted scope 1+2 carbon intensity of the Global Mean Variance Portfolio

print(np.array(portfolios[9]))
print(returns[9])
print(risks[9])

