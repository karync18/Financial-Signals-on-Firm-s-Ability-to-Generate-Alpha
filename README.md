# Financial-Signals-on-Firm-s-Ability-to-Generate-Alpha
This research paper aims to evaluate the effectiveness of five financial signals—firm size, value, operating profitability, firm growth, and momentum—in contribution to a firm’s ability to generate excess (alpha) returns over the period from January 1, 2004 to May 31, 2025.

## Introduction
The pursuit of alpha returns through systemic investment strategies lies at the heart of empirical asset pricing. Alpha returns refer to the excess returns an investment generates above the returns of a benchmark index – in our case, S&P500 index. This project aims to construct a factor-based investment strategy tailored to U.S. equities (excluding financial institutions) by leveraging firm-level characteristics to identify mispriced stocks with alphagenerating potential.

Our strategy integrates five key signals drawn from academic research: Size (market capitalization), Value (book-to-market ratio), Operating Profitability (profitability relative to assets), Firm growth (year-on-year growth in total assets), and Momentum (past returns).

By evaluating performance using key metrics – mean return, Sharpe ratio, Sortino ratio and maximum drawdown, our analysis aims to assess whether a factor-based investment strategy, grounded in value investing and enriched by both fundamental and market-based signals, can systematically generate alpha and mimic key elements of Buffett-style investing in a real-world, out-of-sample setting

## Data Sample
This exploratory study covers the period from 1 January 2004 to 31 May 2025, using financial and market data obtained from WRDS. Our dataset comprises U.S. stocks listed on the NYSE, AMEX, and NASDAQ exchanges.

To ensure comparability and consistency in valuation metrics, financial institution stocks areexcluded from the analysis. This is because financial firms operate under separateregulatory environments and reporting standards frameworks, which make their valuation metrics different and incomparable to non-financial firms.

## Import Pacakges
``` Python
pip install yfinance --upgrade --no-cache-dir

import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sma
import scipy.stats as scs
from scipy.stats.mstats import winsorize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

pd.set_option('display.max_columns', None)

# To prepare the files for output
outfile = open('output.txt', 'w')

#Setting the limit of winsorized tail that is a % of left and right tails
wlimit = 0.025
```

## Compute Annual Stock Returns (Stocks Data csv)
``` Python
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/Colab Notebooks/AVI Group Project/stocks_data_new.csv'

#Store the monthly stocks data into df
df1 = pd.read_csv(file_path, parse_dates=['datadate'])
```

``` Python
#To check the data type for each of the variables
df1.info()

#Convert the iid variable from object datatype to integer
df1['iid'] = pd.to_numeric(df1['iid'], errors='coerce')
df1['iid'] = df1['iid'].astype('Int64')
```
<img width="452" height="314" alt="image" src="https://github.com/user-attachments/assets/30375cb4-676f-4d72-a57e-4c301952ea74" />

``` Python
# Reset the index to replace the left-hand side numbers with 0, 1, 2, ...
df1 = df1.reset_index(drop=True)

#Exclude the companies operating in financial industries (such as banks) by filtering out gsector = 40
df1 = df1[df1['gsector'] != 40]

#To check if gsector = 40 is excluded already from the dataset
#df1['gsector'].value_counts()

#To include only companies listed in NYSE (exchg = 11), american exchange (exchg = 12), and NASDAQ (exchg = 13)
df1 = df1[df1['exchg'].isin([11, 12, 13])]

#To check if the dataframe only include US stocks listed in NYSE, american stock exchange, and NASDAQ
#df1['exchg'].value_counts()

#Sort the dataframe by company's gvkey and datadate
df1 = df1.sort_values(by=['gvkey','datadate'])

#To keep only 1 main class of common stock
df1 = df1[df1['iid'] == 1]
df1.head(20)
```
<img width="730" height="655" alt="image" src="https://github.com/user-attachments/assets/75c15493-686d-4b0a-9f06-b4d9f281c6a3" />

``` Python
#Create a portfolio formation year (pfyr) variable for compounding returns July-June by pfyr
#pfyr = current year for Jul-Dec dates and previous year for Jan-June dates
df1['year'], df1['month'] = df1['datadate'].dt.year, df1['datadate'].dt.month
df1['pfyr'] = np.where(df1.month > 6, df1.year, df1.year - 1)
df1.head(15)

#Copy over df data to df2
df2 = df1.copy()

#Calculate the monthly return compounding factor (1 + monthly return)
df2['mthret1'] = 1 + df2.trt1m/100
df2.describe()
```
<img width="1354" height="277" alt="image" src="https://github.com/user-attachments/assets/5ee2bcd7-54c4-4b38-b5c5-734ca1113cd6" />

## Cleaning Accounting Data CSV
### 1. Import the accounting dataset and sort the data based on gvkey and datadate
``` Python
#The accounting data is retrieved from WRDS using the financial annual north america sample
#There is an option to exclude the stocks in financial indsutries (banks), thus financial industries are already excluded from the dataset

file_path2 = '/content/drive/My Drive/Colab Notebooks/AVI Group Project/accounting_data_new.csv'

#Store the monthly stocks data into df
df3 = pd.read_csv(file_path2, parse_dates=['datadate'])
df3 = df3.sort_values(by=['gvkey','datadate'])
df3.head(20)
```
<img width="1351" height="653" alt="image" src="https://github.com/user-attachments/assets/8cee4814-740e-40fd-80dd-1041ea0bc3f7" />

### 2. Create Portfolio formation year (pfyr) variable
``` Python
#create the portfolio formation year (pfyr) variable where pfyr is current year for Jan-Mar year-end dates and next year for Apr-Dec year-end dates
#Therefore, if fiscal year-end date is Jan-Mar, we will use the info to form the portfolio in July of the current year (pfyr will use the same year)
#However, if the fiscal year-end date is Apr-Dec, we will use the info to form the portfolio in July of the following year (pfyr will use the following year)

df3['year'], df3['month'] = df3['datadate'].dt.year, df3['datadate'].dt.month
df3['pfyr'] = np.where(df3.month < 4, df3.year, df3.year + 1)
df3.head(20)
```
<img width="1493" height="653" alt="image" src="https://github.com/user-attachments/assets/5f91f446-56a2-4d41-bf5b-bd7fa69c471e" />

``` Python
df3.info()
```
<img width="528" height="522" alt="image" src="https://github.com/user-attachments/assets/c0662a91-b169-48bb-9732-bbadbbe2dde3" />

### 3. Observed Signals
To construct our multi-signal investment strategy, we incorporated five signals – three accounting-based and two market-based. Each signal is transformed into a binary indicator based on relative ranking to facilitate portfolio construction.

#### 3a. Create Market Capitalization (for firm size)
We used market capitalization as a proxy for firm size, following Piotroski (2000), whodocumented that smaller firms within the high book-to-market territory tend to generatehigher excess returns, particularly when combined with positive financial signals. This buildson Fama and French (1992), who similarly highlighted the small-cap premium. The rationaleis that small firms are often underestimated or overlooked by investors, allowing those who pay attention to earn better returns. 

To construct the size signal (mc_score), we took the natural logarithm of market capitalization, applied winsorization to minimize the influence of outliers, and ranked firms into deciles. Since market capitalization has an inverse relationship with stock returns, firms in the bottom 30% (i.e., smallest by size) were assigned a score of 1, and 0 otherwise.

``` Python
# Take the natural logarithm of market capitalization (ln of market cap)
df3['ln_market_cap'] = np.log(df3['mkvalt'])
df3.describe()
```
<img width="1072" height="296" alt="image" src="https://github.com/user-attachments/assets/8b6604d6-8369-4d45-aa92-dccab6bb6856" />
<img width="726" height="297" alt="image" src="https://github.com/user-attachments/assets/69f95427-aefd-4b8b-9c05-eb7a1bc81628" />

``` Python
#select the relevant variables and store it inside df4
#remove the infinite values
df4 = df3[['gvkey', 'pfyr', 'ln_market_cap']]
df4 = df4[np.isfinite(df4)]
df4.describe()
```
<img width="392" height="277" alt="image" src="https://github.com/user-attachments/assets/2c95e36d-87ba-4d7d-8c0b-6355796c3018" />

Winsorize Variables
``` Python
#winsorize the variables
df4['market_cap_winsorize'] = winsorize(df4['ln_market_cap'], limits=[wlimit, wlimit])
df4.describe()
```
<img width="561" height="281" alt="image" src="https://github.com/user-attachments/assets/b55298b3-7a40-407f-ac00-d92719e3e593" />

``` Python
#Create and store the relevant variables into a new dataframe
df_market_cap = df4[['gvkey', 'pfyr', 'market_cap_winsorize']]

# Create decile rankings using rank
df_market_cap['mc_win_decile'] = pd.qcut(df_market_cap['market_cap_winsorize'].rank(method='first'), 10, labels=False)

# Sort the DataFrame by the Winsorized 'market_cap' values
df_market_cap_sorted = df_market_cap.sort_values(by=['market_cap_winsorize'], ascending=True)

# take a look st the data
df_market_cap_sorted
```
<img width="458" height="406" alt="image" src="https://github.com/user-attachments/assets/22072f9a-8752-4f95-957a-f37a02f79950" />

Assign score 1 to the smallest 30% of the firms and assign score 0 for the rest inside the mc_win_decile
``` Python
def MC(row):
    return 1 if row['mc_win_decile'] in [0, 1, 2] else 0

# create the scoring column for market_cap
df_market_cap['mc_score'] = df_market_cap.apply(MC, axis=1)
df_market_cap_sorted = df_market_cap.sort_values(by=['mc_score'], ascending=True)

# take a look at the table
df_market_cap_sorted
```
<img width="531" height="400" alt="image" src="https://github.com/user-attachments/assets/efd26530-e6e3-4aad-8dc8-d61590b3d94a" />

Select relevant variables and store them into a new dataframe
``` Python
df_market_cap = df_market_cap[['gvkey','pfyr','mc_score']]
df_market_cap.describe()
```
<img width="393" height="278" alt="image" src="https://github.com/user-attachments/assets/0982f3b3-72f8-42b5-8ddb-d39eb0395bd1" />

#### 3b. Book Market Value of Equity
The book-to-market (BM) ratio is a traditional measure of stock valuation and a proxy to identify undervalued firms. Fama and French (1992) established its predictive power,showing that firms with high BM (value stocks) tend to significantly outperform those with low BM (growth stocks). Piotorski (2000) further refined this by demonstrating that the value premium is driven not equally across all value stocks, but particularly by those with strong financial strength. Combining BM with financial statement analysis can isolate the strongest performers among value firms, especially in small-cap, low-turnover and underfollowed segments. Ahn, Patatoukas, and Skiadopoulos (2024) reaffirmed the book-tomarket ratio as a robust explanatory variable in multifactor models

In our study, the BM ratio is computed dividing the total common or ordinary equity (book value of equity) by the firm’s total fiscal market value (market capitalization), following the formula bv_equity = ceq/mkvalt in our Python code. After winsorizing to mitigate outliers, firms were ranked into declies, and those in top 30% (i.e. the most undervalued) received a score of 1 (bm_score), and 0 otherwise.

``` Python
#Compute the book market value of equity
#BV equity = Total common or ordinary equity/Total fiscal market value
df3['bv_equity'] = df3['ceq']/df3['mkvalt']
df3.describe()
```
<img width="962" height="295" alt="image" src="https://github.com/user-attachments/assets/a95f09bb-69a6-4ed9-b92d-0bfbac1442ba" />
<img width="950" height="293" alt="image" src="https://github.com/user-attachments/assets/c734ea97-9b18-4a86-ae8a-f1f0e152a9f5" />

``` Python
#Cleaning the data by removing the infite values from bv_equity
df5 = df3[['gvkey', 'pfyr', 'bv_equity']]
df5 = df5[np.isfinite(df5)]

#Winsorize the variable
df5['bv_equity_winsorize'] = winsorize(df5['bv_equity'], limits=[wlimit, wlimit])

#Store the relevant variables into a new dataframe
df_bv_equity = df5[['gvkey', 'pfyr', 'bv_equity_winsorize']]
df_bv_equity.describe()
```
<img width="436" height="274" alt="image" src="https://github.com/user-attachments/assets/26df0661-671f-46d6-8b52-ee8d3a27e5d2" />

``` Python
# Use the rank method to create decile rankings
df_bv_equity['bv_eq_win_decile'] = pd.qcut(df_bv_equity['bv_equity_winsorize'].rank(method='first'), 10, labels=False)

# Sort the dataframe using the winsorized bv equity values
df_bv_equity_sorted = df_bv_equity.sort_values(by=['bv_equity_winsorize'], ascending=True)

#Assign score of 1 to the top 30% companies and assign score of 0 for the rest of bv_eq_win_decile
def BV(row):
    return 1 if row['bv_eq_win_decile'] in [7, 8, 9] else 0

df_bv_equity['bv_score'] = df_bv_equity.apply(BV, axis=1)
df_bv_equity_sorted = df_bv_equity.sort_values(by=['bv_score'], ascending=True)

#To check the data
df_bv_equity_sorted.describe()

#Store the relevant variables into a new dataframe
df_bv_equity = df_bv_equity[['gvkey','pfyr','bv_score']]
df_bv_equity.describe()
```
<img width="391" height="280" alt="image" src="https://github.com/user-attachments/assets/1b52e70e-89b4-4bdb-b95f-da8eecb6619b" />

#### 3c. Operating Profitability (Operating Income After Depreciation/BV Equity)
Operating profitability reflects the firm’s ability to generate earnings from core operations. Piotroski (2000) found that stronger operating performance is positively associated with higher future returns – most evident within the value stock universes. These findings suggest that the market often may not recognize the strength of a company’s core business, which creates an opportunity for investor to extend their analysis for superior returns. 

We defined this ratio as Operating Income After Depreciation divided by Book Value of Equity. Noting the ratio’s susceptibility to extreme values, we applied 2.5% winsorization at both tails to make it robust and diminish the effect of outliers. To effectively identify the most profitable firms for portfolio construction, we implemented a binary scoring system, assigning a score of 1 to those in top 30% based on profitability and 0 to the remaining 70%.

``` Python
#Compute the operating profitability
#Operating Profitability = Operating Income After Depreciation/Book Value of Equity
df3['op_profit'] = df3['oiadp']/df3['bv_equity']
df3.describe()
```
<img width="1153" height="295" alt="image" src="https://github.com/user-attachments/assets/ddce1676-2619-4bbb-8ffd-dcf602c43a70" />
<img width="942" height="294" alt="image" src="https://github.com/user-attachments/assets/56e52ba7-350b-4f62-afed-301e92714f8f" />

``` Python
#Cleaning the data by removing the infite values
df6 = df3[['gvkey', 'pfyr', 'op_profit']]
df6 = df6[np.isfinite(df6)]

#Winsorize the variable
df6['op_profit_winsorize'] = winsorize(df6['op_profit'], limits=[wlimit, wlimit])

#Store the relevant variables into a new dataframe
df_op_profit = df6[['gvkey', 'pfyr', 'op_profit_winsorize']]
df_op_profit.describe()
```
<img width="434" height="275" alt="image" src="https://github.com/user-attachments/assets/b6b8d35c-9c25-4df4-98eb-5a661ead1c66" />

``` Python
# Use the rank method to create decile rankings
df_op_profit['op_profit_win_decile'] = pd.qcut(df_op_profit['op_profit_winsorize'].rank(method='first'), 10, labels=False)

# Sort the dataframe using the winsorized op_profit values
df_op_profit_sorted = df_op_profit.sort_values(by=['op_profit_winsorize'], ascending=True)

#Assign score of 1 to the top 30% companies and assign score of 0 for the rest of op_profit_win_decile
def OP(row):
    return 1 if row['op_profit_win_decile'] in [7, 8, 9] else 0 #7, 8, 9 to choose the top 30% of the companies

df_op_profit['op_profit_score'] = df_op_profit.apply(OP, axis=1)
df_op_profit_sorted = df_op_profit.sort_values(by=['op_profit_score'], ascending=True)

#To check the data
df_op_profit_sorted.describe()

df_op_profit = df_op_profit[['gvkey','pfyr','op_profit_score']]
df_op_profit.describe()
```
<img width="415" height="271" alt="image" src="https://github.com/user-attachments/assets/ed759926-b385-4167-be5c-b663be503ebc" />

#### 3d. Investment or Firm's Growth (YoY Growth in Total Assets)
Asset growth serves as a proxy for investment behaviour. In line with Fama and French (2015), low-asset-growth firms tend to outperform their aggressive-growth equivalents. Recent literature has emphasized the importance of growth quality over quantity. For instance, Ahn et al. (2024) showed that growth aligned with material fundamentals is more value-adding than superficial expansion through firms investing in financially material ESG initiatives.

To construct this signal, the data was firstly grouped based on gvkey and then, the firm’s year-over-year percentage change in total assets was computed. After winsorizing, the firms were sorted into deciles. The top 30% (indicative of strategic, high-quality growth) were assigned a score of 1; others received 0. 

``` Python
#Compute the firm's YoY growth by using total assets
#YoY growth in total assets -> use the pct_change formula
df3['yoy_growth_at'] = df3.groupby('gvkey')['at'].pct_change()
df3.describe()
```
<img width="1072" height="293" alt="image" src="https://github.com/user-attachments/assets/533605b5-f545-43eb-8843-845fe094dd3f" />
<img width="1064" height="300" alt="image" src="https://github.com/user-attachments/assets/2c55e885-144b-4c0a-9938-422f128a7028" />

``` Python
#Cleaning the data by removing the infite values from yoy_growth_at
df7 = df3[['gvkey', 'pfyr', 'yoy_growth_at']]
df7 = df7[np.isfinite(df7)]

#Winsorize the variable
df7['yoy_growth_at_winsorize'] = winsorize(df7['yoy_growth_at'], limits=[wlimit, wlimit])

#Store the relevant variables into a new dataframe
df_yoy_growth_at = df7[['gvkey', 'pfyr', 'yoy_growth_at_winsorize']]
df_yoy_growth_at.describe()
```
<img width="469" height="278" alt="image" src="https://github.com/user-attachments/assets/8a6bb033-e0ed-4fc3-951b-8ee41407b58e" />

``` Python
# Use the rank method to create decile rankings
df_yoy_growth_at['yoy_growth_at_win_decile'] = pd.qcut(df_yoy_growth_at['yoy_growth_at_winsorize'].rank(method='first'), 10, labels=False)

# Sort the dataframe using the winsorized yoy_growth_at_winsorize values
df_yoy_growth_at_sorted = df_yoy_growth_at.sort_values(by=['yoy_growth_at_winsorize'], ascending=True)

#Assign score of 1 to the top 30% companies and assign score of 0 for the rest of yoy_growth_at_win_decile
def AT(row):
    return 1 if row['yoy_growth_at_win_decile'] in [7, 8, 9] else 0 #7, 8, 9 to choose the top 30% of the companies

df_yoy_growth_at['yoy_growth_at_score'] = df_yoy_growth_at.apply(AT, axis=1)
df_yoy_growth_at_sorted = df_yoy_growth_at.sort_values(by=['yoy_growth_at_score'], ascending=True)

#To check the data
df_yoy_growth_at_sorted.describe()

df_yoy_growth_at = df_yoy_growth_at[['gvkey','pfyr','yoy_growth_at_score']]
df_yoy_growth_at.describe()
```
<img width="435" height="274" alt="image" src="https://github.com/user-attachments/assets/c6e8646c-65e8-4973-a35f-7480d17cc58c" />

#### 3e. Momentum
Momentum reflects the empirical tendency of asset prices to continue moving in the same direction. It is based on the observation that securities that have performed well (or poorly) in the past are likely to continue their performance trend in the short run, which is commonly used as a predictive signal in asset selection and portfolio construction.

While Piotroski (2000) acknowledged the momentum effect, he also found that fundamental signals deliver distinct and complementary returns. For instance, the study by Lin (2019) supports this by showing that portfolios built based on momentum yield positive returns that cannot be adequately explained by traditional asset pricing models.

To construct the momentum signal, we computed each firm’s cumulative log return over the prior 11 months, excluding the most recent month to mitigate short-term reversal effects, using a one-month-lagged rolling sum of monthly log returns. This rolling measure is then winsorized and firms are ranked into deciles. Those in the top 30% were given a score of 1 (momentum_score), with all others receiving 0. Returns are measured from July to June in line with standard momentum practices.

``` Python
#Keep only the relevant variables that we are going to use
#Sort the data by 'gvkey' and 'pfy'
df2 = df2.sort_values(by=['gvkey', 'pfyr'])
df8 = df2[['gvkey', 'tic', 'conm', 'datadate', 'month', 'pfyr', 'mthret1']]
df8 = df8.dropna()

#This is to calculate the continuously compounded monthly return
df8['log_return'] = np.log(1 + df8['mthret1'])

#Use a function to calculate the momentum
#Then, we apply the function to each of the 'gvkey' group
#Reset the existing index
def calculate_momentum(df):
    df = df.sort_values(by='pfyr')
    df['momentum'] = df['log_return'].rolling(window=11).sum().shift(1)
    return df

df8 = df8.groupby('gvkey').apply(calculate_momentum).reset_index(drop=True)

# We want to remove the 'log_return' column and remove rows with NaN values in the 'momentum' column
df8 = df8.drop(columns=['log_return'])
df8 = df8.dropna(subset=['momentum'])

# Form the momentum_decile portfolio and assign score of 1 to the top 30% and 0 to the remaining data
df8['momentum_decile'] = pd.qcut(df8['momentum'], 10, labels=False)
df8['momentum_signal_score'] = np.where(df8['momentum_decile'] >= 7, 1, 0)

# To make the data appear
df8
```
<img width="962" height="399" alt="image" src="https://github.com/user-attachments/assets/31727538-0c6e-4c88-b775-a3c49c3cfab3" />

``` Python
# Group by 'gvkey' and 'pfyr' to calculate the annual returns for each pfyr
# Afterwards, we want to take only companies that have data of 12 months return from July to June
df8['yret'] = df8.groupby(['gvkey', 'pfyr'])['mthret1'].cumprod() - 1
df9 = df8.groupby(['gvkey', 'pfyr']).nth(11) # Take the 12th element of each pfyr for each of the firm

#Winsorize the return
df9['yret_winsorize'] = winsorize(df9['yret'], limits=[wlimit, wlimit])

#To see the summary of the data (mean, max, min, std dev, etc)
df9.describe()
```
<img width="1328" height="280" alt="image" src="https://github.com/user-attachments/assets/e35b7a19-71a9-43ca-abdd-b34416d03a44" />

``` Python
df9.info()
```
<img width="532" height="331" alt="image" src="https://github.com/user-attachments/assets/0317b440-3cc6-43e9-8741-d566eda17ff3" />

### 4. Merge Dataframes with Signal Scores
In this section, we want to select the relevant variables and merge it with the signal scores

``` Python
# List of dataframes that we want to merge
dfs = [df9, df_market_cap, df_bv_equity, df_op_profit, df_yoy_growth_at]

# Define a function to merge two DataFrames based on 2 variables which are ['gvkey', 'pfy']
def merge_dfs(left, right):
    return pd.merge(left, right, on=['gvkey', 'pfyr'], how='left')

final_df = reduce(merge_dfs, dfs) #use reduce to apply the function to the list of df

cols = list(final_df.columns) # Get the list of columns
cols.insert(0, cols.pop(cols.index('conm'))) #rearrange the column position

final_df = final_df[cols]

#to remove duplicates and NA's in the final dataframe
final_df = final_df.drop_duplicates(subset=['gvkey', 'pfyr','month'], keep='first')
final_df = final_df.dropna()

#To show the final dataframe result with numbers rounded to whole number
final_df.describe().round(0)
```
<img width="1548" height="285" alt="image" src="https://github.com/user-attachments/assets/dd11ac0a-b22b-4c74-a62a-3bc8dde11e54" />

### 5. Model Structure
In our analysis, we employed linear regression to evaluate the relationship between annual stock returns and the five key predictive signals. Each signal is converted into a binary scoreby ranking firms into deciles and assigning a score of 1 to those in the top 30%, and 0 otherwise. By regressing winsorized annual returns on these binary signal indicators, we assessed whether the selected signals possess predictive power in explaining cross-sectional variation in stock returns and generating alpha returns.

``` Python
# Perform t test to see if if the high signal portfolio outperforms the baseline portfolio
t_test_result = scs.ttest_rel(df_high['yret_high'], df_baseline['yret_baseline'])

t_test_df = pd.DataFrame({
    'statistic': [t_test_result.statistic],
    'pvalue': [t_test_result.pvalue],
    'df': [len(df_high) - 1]  # Degrees of freedom: n - 1
})

t_test_df
```
<img width="233" height="64" alt="image" src="https://github.com/user-attachments/assets/9933db34-2914-4f3a-a785-b2cebe82174f" />


### 6. Annual Return on 10 Decile Portfolios
Based on the regression result of annual returns on the five signals, four out of five signals demonstrated outcomes consistent with our expectations. However, the result for the market capitalization (firm size) signal deviated from the anticipated direction, indicating an inverse relationship compared to what was predicted.

#### 6a. Market Capitalization (for firm size)
``` Python
# Group the dataframe by market capitalization score (mc_score) and compute the count and mean 
# for both mc_score and the winsorized yearly return (yret_winsorize)
final_df.groupby('mc_score')[['mc_score','yret_winsorize']].agg(['count', 'mean'])
```
<img width="305" height="155" alt="image" src="https://github.com/user-attachments/assets/f9513c83-d83b-48b8-9f4b-fcf2f28d093f" />

**Explanation:**

Based on the results, firms with higher market capitalization (big size firms) generated an average return of 10.6%, outperforming the bottom 30% (small-cap firms), which earned9.4%. This finding is contrary to our initial assumption that smaller firms typically generate excess returns

#### 6b. Book Market Value of Equity
``` Python
# Group the dataframe by Book Value score (bv_score) and compute the count and mean 
# for both bv_score and the winsorized yearly return (yret_winsorize)
final_df.groupby('bv_score')[['bv_score','yret_winsorize']].agg(['count', 'mean'])
```
<img width="300" height="155" alt="image" src="https://github.com/user-attachments/assets/65dabcf4-43d9-4531-8ab7-837b3de59f6d" />

**Explanation:**
Based on the results, firms with higher book-to-market (BM) ratio – classified as value stocks – achieved a higher mean return of 14.5%, compared to 9.8% for the bottom 70% (low BM or growth stocks).

#### 6c. Operating Profitability (Operating Income After Depreciation/BV Equity)
``` Python
# Group the dataframe by operating profit score (op_profit_score) and compute the count and mean 
# for both op_profit_score and the winsorized yearly return (yret_winsorize)
final_df.groupby('op_profit_score')[['op_profit_score','yret_winsorize']].agg(['count', 'mean'])
```
<img width="392" height="163" alt="image" src="https://github.com/user-attachments/assets/279d836d-e27e-487d-afd7-5a3b3932e7d6" />

**Explanation:**
Based on the results, firms with higher operating profitability (top 30%) delivered an average return of 11.8%, outperforming the bottom 30%, which generated only 9.6%. This indicates a positive relationship between operating profitability and stock returns.

#### 6d. Investment or Firm's Growth (YoY Growth in Total Assets)
``` Python
# Group the dataframe by year on year growth on total asset score (yoy_growth_at_score) and compute the count and mean 
# for both yoy_growth_at_score and the winsorized yearly return (yret_winsorize)
final_df.groupby('yoy_growth_at_score')[['yoy_growth_at_score','yret_winsorize']].agg(['count', 'mean'])
```
<img width="453" height="161" alt="image" src="https://github.com/user-attachments/assets/6e0e5db4-876b-423e-8a98-400d308e4f83" />

**Explanation:**
Interestingly, firms in the bottom 70% for year-over-year asset growth posted a higher average annual return of 10.7%, compared to the fastest-growing top 30% firms, which delivered a lower return of 9.9%.

#### 6e. Momentum
``` Python
# Group the dataframe by momentum signal score (momentum_signal_score) and compute the count and mean 
# for both momentum_signal_score and the winsorized yearly return (yret_winsorize)
final_df.groupby('momentum_signal_score')[['momentum_signal_score','yret_winsorize']].agg(['count', 'mean'])
```
<img width="490" height="157" alt="image" src="https://github.com/user-attachments/assets/31d153dd-20d9-4074-820f-faba269c074d" />

**Explanation:**
The results show a pronounced return disparity between high- and low-momentum firms.Firms in the top 30% of momentum cohort achieved an average annual return of 49.3%, while those in the lower 70% experienced a negative return of -6.3%. This provides robust empirical evidence for the momentum effect and highlights its effectiveness as a return-enhancing strategy, even after adjusting for outliers through winsorization.

## Portfolio Analysis
### 1. Final Composite Score

``` Python
# Calculate total signal score by summing across selected columns
signal_columns = [
    'momentum_signal_score',
    'mc_score',
    'bv_score',
    'op_profit_score',
    'yoy_growth_at_score'
]

# Create the 'total_score' column
final_df['total_score'] = final_df[signal_columns].sum(axis=1)

# Group by total_score and calculate count and mean of annual returns
score_summary = final_df.groupby('total_score')['yret_winsorize'].agg(['count', 'mean'])

# Display result
print(score_summary)
```
<img width="229" height="125" alt="image" src="https://github.com/user-attachments/assets/73ad61a4-ef34-4446-b73c-b21a891d742b" />

**Explanation:**
The table below presents the total scores derived from our five investment signals; market capitalisation, book-to-market value, operating profitability, asset growth, and momentum.

The table shows a strong positive relationship between the total_score and average annual returns. Firms with a score of 0 earned a negative average return of –1.5%, while those with the the highest score of 4 achieved an impressive average return of nearly 59%. As the totalscore increases, the number of firms decreases, but the average return improves. Therefore,it illustrates that the higher total_scores are associated with stronger future performance.This pattern supports the effectiveness of total_score as a predictive signal for stock returns.

### 2. Variable Analysis
#### 2a. Correlation Analysis
``` Python
#Filter out low total_score values
final_df_cleaned = final_df[(final_df['total_score'] > -3)]
final_df_cleaned.describe().round(0)
```
<img width="1790" height="316" alt="image" src="https://github.com/user-attachments/assets/f94eed27-5068-47ce-816e-3999a403481c" />

**Pearson Corrlation**

``` Python
# Correlation Coefficients (Pearson)
final_df_cleaned[['momentum_signal_score','mc_score','bv_score','op_profit_score','yoy_growth_at_score']].corr()
```
<img width="797" height="182" alt="image" src="https://github.com/user-attachments/assets/978b447b-511f-4916-8ee8-49f8f3805a01" />

**Spearman Correlation:**

``` Python
# Correlation Coefficients (Spearman)
final_df_clean[['momentum_signal_score','mc_score','bv_score','op_profit_score','yoy_growth_at_score']].corr(method="spearman")
```
<img width="792" height="181" alt="image" src="https://github.com/user-attachments/assets/acefabce-6173-412f-ac6c-f3843cde24b3" />

**Explanation:**
Based on the two tables above, the momentum signal score and four fundamental factors consistently illustrated weak relationships in both the Pearson and Spearman correlation matrices. Specifically, momentum has low positive correlations with market capitalization (4%), book value (8.7%), operating profitability (7%), and year-over-year growth in total assets (5.3%). These findings suggest that the momentum signal is largely independent of traditional fundamental indicators. Moreover, the close alignment between Pearson and Spearman coefficients indicates that the relationships are approximately linear and monotonic, with no significant influence from outliers or non-linear patterns.

Among the fundamentals, there are a few notable relationships:
- Book value and market capitalization are moderately correlated (0.208), reflecting their common link to firm size.
- Market capitalization and operating profitability show a negative correlation (–0.193), possibly indicating differing profitability patterns across firm sizes.
- Operating profitability and asset growth are positively correlated (0.115), suggesting that more profitable firms tend to grow more rapidly.

#### 2b. Regression Analysis

**Multi-signal OLS Regression (All 5 Signals + Total Score)**
``` Python
# Define the OLS regression function
def olsreg(d, yvar, xvars):
    Y = d[yvar]
    X = sma.add_constant(d[xvars])
    reg = sma.OLS(Y, X).fit()
    return reg.params

# Group by year and run regression on each group
final_df_clean_group = final_df_clean.groupby('pfyr')
yearcoef = final_df_clean_group.apply(olsreg, 'yret_winsorize',
                                       ['momentum_signal_score', 'mc_score', 'bv_score',
                                        'op_profit_score', 'yoy_growth_at_score', 'total_score'])

# Run t-test to see if average coefficient is significantly different from 0
tstat, pval = scs.ttest_1samp(yearcoef, 0)

# Display results
print("\n T-Test Results for All Signals:")
print("t-statistics:\n", tstat)
print("p-values:\n", pval)
```
<img width="573" height="127" alt="image" src="https://github.com/user-attachments/assets/858b391f-47b8-4eee-92ea-e1b63c540648" />

**Single-Variable Regression (Total Score Only)**
``` Python
# Define separate function (optional, same logic as above)
def olsreg_total(d, yvar, xvar):
    Y = d[yvar]
    X = sma.add_constant(d[[xvar]])
    reg = sma.OLS(Y, X).fit()
    return reg.params

# Apply regression by year
yearcoef_total = final_df_clean_group.apply(olsreg_total, 'yret_winsorize', 'total_score')

# T-test for average coefficient
tstat_total, pval_total = scs.ttest_1samp(yearcoef_total, 0)

# Display results
print("\n T-Test Results for Total Score Only:")
print("t-statistics:\n", tstat_total)
print("p-values:\n", pval_total)
```
<img width="284" height="96" alt="image" src="https://github.com/user-attachments/assets/d3730132-f57a-413d-a1b6-ad57d4459699" />

**Explanation:**

In the OLS regression of annual stock returns on the selected composite signal (total_score), the results show a t-statistic of 10.12223859 and a p-value of 7.40547243e-09. This indicatesthat total_score has a statistically significant positive relationship with annual stock returns, as the p-value is below the 0.05 significance threshold. The findings suggest that firms with higher fundamental scores deliver higher annual returns, which reinforces the value of total_score as a predictive and reliable signal for stock performance.

### 3. Form Final Portfolio

To evaluate the performance of the signal, firms were split into three group based on their total signal score: low, medium, and high signal portfolios. For comparison, we defined the “high” portfolio as firms in the top tercile (i.e., those with total_score_decile = 2), while the bottom two terciles (i.e., total_score_decile ≤ 1) were combined to serve as the baseline portfolio.

``` Python
# Compute annual returns of decile portfolios 
# Annual returns on 3 groups portfolios sorted year-by-year on on tot_score
final_df['total_score_decile'] = pd.qcut(final_df['total_score'], 3, labels=False)
final_df.groupby('total_score_decile')[['total_score','yret_winsorize']].agg(['count', 'mean'])

final_df.to_csv('final_portfolio.csv', index=False)
```

``` Python
# Step 2: Split final into portfolio to high and baseline of portfolio
df_high = final_df[(final_df['total_score_decile'] == 2)]
df_high = df_high.rename(columns={'yret_winsorize': 'yret_high'})[['pfyr', 'yret_high']]

df_baseline = final_df[(final_df['total_score_decile'] <= 1)]
df_baseline = final_df.rename(columns={'yret_winsorize': 'yret_baseline'})[['pfyr', 'yret_baseline']]
```

``` Python
df_high
```
<img width="176" height="380" alt="image" src="https://github.com/user-attachments/assets/57db9d45-6c91-41a7-b556-85860b6219b4" />

``` Python
df_high.describe()
```
<img width="248" height="282" alt="image" src="https://github.com/user-attachments/assets/7181ac89-3675-4306-b8cc-ec1981919d4b" />

``` Python
df_baseline
```
<img width="169" height="656" alt="image" src="https://github.com/user-attachments/assets/f75cbd23-3ff2-4dac-af1a-5296edcdbb50" />

``` Python
df_baseline.describe()
```
<img width="266" height="280" alt="image" src="https://github.com/user-attachments/assets/0bf5a7f5-8f4f-4570-ba38-c18ffd8883c6" />

**Annual Returns Over Time**
``` Python
import matplotlib.pyplot as plt

# Define the data
x = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
     2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

y = df_high['yret_high'].values[:len(x)]
z = df_baseline['yret_baseline'].values[:len(x)]

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data with different colors and markers
ax.plot(x, y, color='blue', marker='o', label='Portfolio to Long')
ax.plot(x, z, color='green', marker='s', label='Rest of Portfolio')

# Labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Annual Return')
ax.set_title('Annual Returns Over Time')
ax.legend()

# Rotate x-axis labels for readability
ax.set_xticks(x)
ax.tick_params(axis='x', rotation=90)

# Save and show the plot
fig.savefig('portfolio_returns_simple.png')
plt.show()
```
<img width="706" height="556" alt="image" src="https://github.com/user-attachments/assets/c5290c74-3c93-431f-9c05-d5e1e08e70e2" />

## Performance Analysis
To assess performance, we compared the high portfolio with the baseline portfolio using their Sharpe and Sortino Ratios, along with maximum drawdown metrics.

### 1. Sharpe Ratio
``` Python
# Computing Sharpe Ratio
def compute_sharpe_ratio(returns, label):
    sharpe_ratio = returns.mean() / returns.std()
    print(f"Sharpe Ratio for {label}: {sharpe_ratio:.4f}")
    return sharpe_ratio

# Compute Sharpe ratios for two portfolios
sharpe_high = compute_sharpe_ratio(df_high['yret_high'], "High Portfolio")
sharpe_baseline = compute_sharpe_ratio(df_baseline['yret_baseline'], "Baseline of Portfolio")
```
<img width="359" height="38" alt="image" src="https://github.com/user-attachments/assets/32dc513f-bb79-4488-a78e-bc081402d0d4" />

### 2. Sortino Ratio
``` Python
#Calculate the sortino ratio
import numpy as np

def compute_sortino_ratio(returns, target=0.0):
    """
    Compute Sortino Ratio using the same logic as the original version:
    - Use all returns (mean of full sample)
    - Only penalize returns below the target (with squared deviation)
    """
    # Calculate excess return
    excess_return = returns - target

    # Downside deviation (square of only negative excess returns)
    squared_downside = np.minimum(0, excess_return) ** 2
    downside_deviation = np.sqrt(np.mean(squared_downside))  # mean over full sample

    # Sortino Ratio
    sortino_ratio = excess_return.mean() / downside_deviation if downside_deviation != 0 else np.nan

    return sortino_ratio

# Apply to both portfolios
sortino_high = compute_sortino_ratio(df_high['yret_high'], target=0)
print(f"Sortino Ratio (Long Portfolio): {sortino_high:.4f}")

sortino_baseline = compute_sortino_ratio(df_baseline['yret_baseline'], target=0)
print(f"Sortino Ratio (Rest of Portfolio): {sortino_baseline:.4f}")
```
<img width="324" height="43" alt="image" src="https://github.com/user-attachments/assets/a8a0efe0-59f0-45c1-8388-0a2a2d5487b6" />


### 3. Maximum Drawdown
``` Python
# Group yearly returns by portfolio formation years
df_high = df_high.groupby('pfyr').mean()
df_baseline = df_baseline.groupby('pfyr').mean()
```

**Maximum Drawdown for portfolio to high and baseline portfolio**
``` Python
# Compute maximum drawdown

cumret = np.cumprod(df_high['yret_high']) 
hwm = cumret.cummax()
drawdown_perc = ((hwm - cumret) / hwm) * 100
drawdown_abs = hwm - cumret

maxDD_perc = drawdown_perc.max()
maxDD_abs = drawdown_abs.max()

print('Maximum drawdown for portfolio to high = ', round(maxDD_abs,2))
print('Maximum drawdown for portfolio to high in % = ', round(maxDD_perc,2))


cumret1 = np.cumprod(df_baseline['yret_baseline']) 
hwm1 = cumret1.cummax()
drawdown_perc1 = ((hwm1 - cumret1) / hwm1) * 100
drawdown_abs1 = hwm1 - cumret1

maxDD_perc1 = drawdown_perc1.max()
maxDD_abs1 = drawdown_abs1.max()

print()
print('Maximum drawdown for baseline of portfolio = ', round(maxDD_abs1,2))
print('Maximum drawdown for baseline of portfolio in % = ', round(maxDD_perc1,2))
```
<img width="446" height="85" alt="image" src="https://github.com/user-attachments/assets/b73cd946-5960-4f2d-b895-5b11ed6c5f1b" />

**Maximum drawdown for long portfolio and the rest of portfolio**
``` Python
import numpy as np

def compute_max_drawdown(returns, label="Portfolio"):
    """
    Computes the maximum drawdown (absolute and percent) from a series of returns.
    
    Args:
        returns (pd.Series): Series of yearly returns (in decimal form, e.g., 0.12 for 12%)
        label (str): Optional label for printout.
        
    Returns:
        (float, float): Tuple of (max drawdown in absolute units, in %)
    """
    cumulative = np.cumprod(1 + returns)
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    drawdown_pct = (drawdown / peak) * 100

    max_abs_dd = drawdown.max()
    max_pct_dd = drawdown_pct.max()

    print(f"Maximum drawdown for {label}: {max_abs_dd:.2f}")
    print(f"Maximum drawdown for {label} in %: {max_pct_dd:.2f}%\n")

    return max_abs_dd, max_pct_dd

# Compute for both portfolios
maxDD_abs_high, maxDD_perc_high = compute_max_drawdown(df_high['yret_high'], label="Long Portfolio")
maxDD_abs_baseline, maxDD_perc_baseline = compute_max_drawdown(df_baseline['yret_baseline'], label="Rest of Portfolio")
```
<img width="402" height="89" alt="image" src="https://github.com/user-attachments/assets/9ae1c7b5-db62-4618-a4dc-9a862363f9a1" />

**Performance Analysis Explanation:**
The high portfolio generated an impressive average annual return of 46%, compared to just 10.5% for the baseline portfolio – an outperformance of nearly fourfold. This suggests that a systematic approach using the five combined signals can help investors to improve stock selection.

In terms of risk-adjusted returns, the high portfolio stands out again. It achieved a Sharpe Ratio of 1.128, which is substantially higher than the baseline portfolio’s 0.293, indicating that the high portfolio earns significantly more return per unit of total risk. Moreover, the Sortinoratio which focuses more on the downside risk, paints an even clearer picture. The high portfolio recorded a Sortino Ratio of 4.35, compared to just 0.61 for the baseline portfolio. This indicates that the high-score portfolio not only generated stronger returns, but did so with considerably less downside volatility, making it an attractive option from risk perspective.

We also evaluated how each portfolio performed during periods of market stress using the Maximum Drawdown metric, which captures the largest decline from peak to trough. The high portfolio experienced a drawdown of just 4.08%, while the baseline portfolio suffered a major 26.71% drawdown. In absolute terms, the high portfolio’s drawdown is 0.3714, compared to 0.1591 for the baseline. These findings underlined the high portfolio’s greater resilience to market decline and superiority in preserving capital.


## Portfolio Performance Against S&P 500
``` Python
pip install yfinance --upgrade --no-cache-dir
pip install yfinance pandas matplotlib

import yfinance as yf
import pandas as pd

# Download S&P 500 data (adjusted close) from 2005-01-01 to 2023-12-31
data = yf.download('^GSPC', start='2005-01-01', end='2023-12-31')

# Calculate daily returns
daily_returns = data['Close'].pct_change()

# Calculate annual returns by compounding daily returns for each year
annual_returns = (1 + daily_returns).resample('Y').prod() - 1

# Change index to year only
annual_returns.index = annual_returns.index.year

# Print annual returns rounded to 4 decimals
print("S&P 500 Annual Returns (2005-2023):")
print(annual_returns.round(6))
```
<img width="1378" height="478" alt="image" src="https://github.com/user-attachments/assets/8dd0a23c-2b60-44b5-927c-d2be864c74af" />

**Plotting Annual Returns Over Time**
``` Python
import matplotlib.pyplot as plt

# Define the years
x = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
     2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Make sure the arrays are aligned in length
y = df_high['yret_high'].values[:len(x)]
z = df_baseline['yret_baseline'].values[:len(x)]
sp500 = annual_returns.values[:len(x)]  # Assuming annual_returns is a pandas Series aligned to x

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each series with distinct color and marker
ax.plot(x, y, color='blue', marker='o', label='Portfolio to Long')
ax.plot(x, z, color='green', marker='s', label='Rest of Portfolio')
ax.plot(x, sp500, color='red', marker='^', label='S&P 500')

# Set axis labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Annual Return')
ax.set_title('Annual Returns Over Time')

# Show legend
ax.legend()

# Rotate x-axis ticks for better readability
ax.set_xticks(x)
ax.tick_params(axis='x', rotation=90)

# Save and show plot
fig.savefig('portfolio_returns_comparison.png')
plt.show()
```
<img width="712" height="558" alt="image" src="https://github.com/user-attachments/assets/e6c2fc9f-370d-4459-a626-16ff909baca3" />

**Explanation:**

We assessed whether the high portfolio consistently outperformed and generated alpha returns above and beyond what would be expected given market exposure. Overall, the high portfolio demonstrates strong and consistent outperformance. It not only generated higher average annual returns but also captured more of the upside during bullish markets while displaying greater resilience during downturns.

Almost every year except 2019 and 2021, the high portfolio outpaced the S&P 500 significantly, consistently capturing more of the market's gains. In 2019, the Federal Reserve implemented interest rate cuts three times, which made borrowing cheaper and encouraged more spending and investing. This helped companies, especially in the fintech sectors, do well. In addition, the easing tensions between the U.S. and China toward the end of the year, including progress on a "Phase One" trade deal, further boosted confidence in global markets. Since the S&P 500 includes many of these companies and the high portfolio did not include financials, the S&P 500 ended up doing better that year.

In 2021, the market was driven by an extraordinary post-pandemic recovery, underpinned by massive fiscal stimulus and ultra-loose monetary policy. The U.S. government introduced massive stimulus packages, and the Federal Reserve kept interest rates near zero. This injection of liquidity propelled equity markets, especially for large-cap growth stocks. Tech giants like Apple, Microsoft, Tesla, and NVIDIA – heavily weighted in the S&P 500 index –

## Research Limitation
There are several limitations that our group faced conducting this research. Firstly, our research is limited to only 5 signals to identify mispriced stocks with alpha-generating potential, such as ESG scores. However, our group found out that the ESG data availability is limited to August 2009–September 2015, and thus we decided to exclude the variable from our research. Other signals to consider would be default risk, operating stability, cheapness, and earnings quality. 

Secondly, this study is limited to U.S. companies listed on the NYSE, American Stock Exchange, and NASDAQ only for the period of January 2004 to May 2025. While this sample meets the research requirements, other researchers are encouraged to explore alternative subsets of the tradeable U.S. equity universe, excluding banks and financial institutions, that also satisfy the minimum data coverage and relevance criteria. For instance, the usage of different time frames, market capitalization thresholds, or index constituents, depending on the specific research objectives. 

## Conclusion
Based on our linear regression analysis of annual returns on five financial signals, our findings indicated that four out of five signals aligned with our expectations. The only exception wasthe market capitalization signal, which measures firm size. Contrary to our hypothesis,smaller firms did not consistently generate excess returns compared to larger firms. Furthermore, our analysis shows that the high-ranked portfolios consistently outperformed the market, delivering alpha returns above and beyond the S&P 500 benchmark. This suggests that,despite certain limitations, such as the restricted number of signals and the limited tradeable universe of U.S. companies, the selected variables effectively contributed to identifying firms with strong return potential. 

We believe this research offers valuable insights for investors seeking to evaluate financial variables with a positive impact on firm performance, as well as for students conducting similar studies. Future research may benefit from expanding the signal set or exploring alternative samples within the tradeable U.S. equity universe, provided they meet the necessary data and methodological requirements













