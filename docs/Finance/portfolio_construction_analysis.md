# Fundamentals of Risk and Returns

### Measures of Risk and Reward

We look at its excess return over the risk free rate. 
$$
sharpe \ ratio= \frac{r - r_f}{\sigma} 
$$

### Max Drawdown

The max drawdown is the max loss from the previous high to a subsequent low. The worst possible return you could have. 

The Calmar ratio is the ratio of the annualized return over the trailing 36 months to the max drawdown over those periods of time
$$
calmar \ ratio = \frac{r}{max \ drawdown}
$$
It depends only on two data points, so it is extremely sensitive. Also, it depedends on the granularity of the data (daily/weekly/monthly). 

### Deviations from Normality

Skewness

The shift in mean. 
$$
S(r) = \frac{E[(r - E[r])^3]}{(Var[r])^{3/2}}
$$
Kurtotic

The variance at each point compared to the normal curve. Normal K is 3. 
$$
K(r) = \frac{E[(r - E[r])^4]}{(Var[r])^{2}}
$$
Jarque Bera test
$$
JB = \frac n 6(s^2+\frac{(K-3)^2}{4})
$$
And then do chi-square test (2).

### Downside Risk Measures

Semi-deviation is the volatility of the sub-sample of below-average or below-zero returns. 
$$
\sigma_{semi} = \sqrt{\frac 1 N \sum_{r_t < \bar r}(r_t - \bar r)^2}
$$
Value at risk (VaR) represents the max expected loss over a give time period (e.g., one month) at a specified confidence level (e.g., 99%). Typically expressed as a positive number. 

Expected loss beyond VaR, also called conditional VaR (CVaR).

### Estimating VaR

#### Historical Methodology

Calculation based on the dist of historical changes in the val of the current portfolio under mkt prices over the specified historical observation window. 

#### Parametric Gaussian Methodology

Calc based on portfolio volatility. 

#### Parametric Non-Gaussian Methodology

#### Cornish-Fisher VaR

An alternative to parametric exists semi-parametric approach. 

