# Single Period Mean-Variance Optimization (MVO, Markowitz) with scope 1+2 carbon intensity constraints

## Python script for running a single period mean variance optimization (Markowitz, 1952) with a weighted portfolio scope 1+2 carbon intensity target on top of the "usual" long-only constraints and having the portfolio weights sum up to 1. Carbon intensity, or carbon emissions per dollar of revenue, adjusts for company size and is generally accepted to be a more accurate measurement of the efficiency of output rather than a portfolio's absolute carbon footprint.

NB - Markowitz's Modern Portfolio Theory assumes (amongst others):

- frictionless markets
- market liquidity is infinite
- investors are risk averse 
- returns are normally distributed

Also, kindly note that:

- this script is highly time period sensitive - whilst it also considers the mean daily return to be a good estimator of future returns
- the Quandl module isn't actively supported anymore for returns post April 11, 2018

This script requires the following packages / modules in order to function properly:

- [Python 3.6.5](https://www.python.org/downloads/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Quandl](https://www.quandl.com/databases/WIKIP)
- [CVXOPT](http://cvxopt.org/)

#### Screenshot - Efficient Frontier
![alt text](https://github.com/Weesper1985/Mean_Variance_Portfolio_Optimization_with_Carbon_Intensity_Constraints/blob/master/efficient_frontier.png)

#### Screenshot - Portfolio Breakdown - Global Min Variance Portfolio
![alt text](https://github.com/Weesper1985/Mean_Variance_Portfolio_Optimization_with_Carbon_Intensity_Constraints/blob/master/Breakdown.png)


