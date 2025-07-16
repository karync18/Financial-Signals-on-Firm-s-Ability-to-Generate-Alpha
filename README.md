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









