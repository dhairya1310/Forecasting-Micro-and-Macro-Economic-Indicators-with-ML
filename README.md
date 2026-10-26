# Forecasting Macro and Microeconomic Indicators with ML

**Can classical machine learning models and LSTMs forecast inflation, national income, and equity markets, and where do they break down?**

This project applies Linear Regression, Decision Trees, Random Forests, and Long Short Term Memory (LSTM) networks to three forecasting problems. The first is India's consumer price inflation. The second is GDP per capita across 227 countries. The third is the daily level of the S&P 500.

Tree based models captured cross country income patterns that linear models missed. The inflation and equity results show something just as important. A high score alone does not prove that a model can forecast. Each result below is reported along with what it does and does not mean.

This research was published and presented at the **Hinweis International Conference ASIT 2024**.

---

## Results at a Glance

| Track | Data | Best model | Test score |
|---|---|---|---|
| Inflation | India CPI All India Index, 2013 to 2020, 28 component sub indices | Random Forest | R² 0.994 |
| GDP | GDP per capita across 227 countries, 18 socioeconomic features | Random Forest | R² 0.957 |
| Equity index | S&P 500, daily OHLC, 2016 to 2020 | LSTM, 10 day window | 0.1% to 1.4% error on a 4 day holdout |

---

## Why This Project

Central banks, governments, and investors all depend on forecasts of inflation, growth, and markets. Traditional econometric models assume linear relationships and stable conditions. Machine learning can capture nonlinear patterns, but it can also produce scores that look impressive and mean very little.

This project had two goals.

1. Compare classical machine learning models against deep learning on real economic and financial data
2. Test whether the strong scores actually reflect forecasting ability, or something simpler

---

## Track 1. Inflation Forecasting (India CPI)

### Data
- India's Consumer Price Index "All India Index" published monthly from 2013 to 2020
- Separate series for Rural, Urban, and Combined sectors
- 28 sub indices including cereals, meat and fish, milk, oils, vegetables, fuel and light, clothing, housing, health, transport, and education
- Target variable is the **General Index**

### Approach
1. Cleaned missing values and aligned all sub indices by month and sector
2. Measured how strongly each sub index moves with the General Index. Most correlations were above 0.97
3. Split the data into training and test sets
4. Trained Linear Regression, a tree based model, and a Random Forest regressor
5. Evaluated each model with R² on the test set

### Results

| Model | Test R² |
|---|---|
| Linear Regression | 0.986 |
| Tree based model | 0.993 |
| Random Forest | 0.994 |

### What the Result Means
Every model scored above 0.98, including simple linear regression. The reason is structural. The CPI General Index is calculated as a weighted average of the same sub indices used as inputs.

```
General Index ≈ w1 × Cereals + w2 × Fuel + w3 × Clothing + ...
```

The models were reconstructing a known formula rather than forecasting the future. This is a form of **target leakage**, where the inputs contain the answer. The track is still useful because it shows which components move most closely with headline inflation. It is not a true inflation forecast.

---

## Track 2. GDP per Capita Across Countries

### Data
- A cross country dataset covering 227 countries
- 18 features including population, area, population density, coastline, net migration, infant mortality, literacy, phones per 1,000 people, arable land, climate, birth rate, death rate, and the share of agriculture, industry, and services in the economy
- Target variable is **GDP per capita in US dollars**

### Approach
1. Handled missing values and converted text formatted numbers to numeric values
2. Built a correlation heatmap and scatter plots of GDP against each feature
3. Split the data into training and test sets
4. Trained Linear Regression, a Decision Tree, and a Random Forest regressor
5. Evaluated with R² and Root Mean Squared Error (RMSE)

### Strongest Relationships

| Feature | Correlation with GDP per capita |
|---|---|
| Phones per 1,000 people | 0.83 |
| Birth rate | −0.64 |
| Infant mortality | −0.60 |
| Agriculture share of economy | −0.59 |
| Service share of economy | 0.55 |
| Literacy | 0.51 |

### Results

| Model | Test R² | RMSE |
|---|---|---|
| Linear Regression | 0.36 | $6,131 |
| Decision Tree | 0.37 | $6,076 |
| Random Forest | 0.957 | |

### What the Result Means
Linear Regression performed poorly because the relationships are not straight lines. GDP per capita stays low across most literacy levels and then rises sharply once literacy passes roughly 80%. Infant mortality shows the reverse pattern. Tree based models handle thresholds like these naturally, which explains why they outperformed the linear model.

Two cautions apply.

1. The jump from 0.37 for a single tree to 0.957 for a Random Forest is unusually large. With only 227 countries, one train and test split can swing scores a lot, and this result should be confirmed with cross validation.
2. The dataset is a single snapshot in time. It explains why countries differ from one another, such as why Norway is richer than Niger. It does not forecast how a country's GDP will change next year.

---

## Track 3. S&P 500 Forecasting with LSTM

### Data
- Daily High, Low, Open, and Close values of the S&P 500 from 2016 to 2020
- The period includes the 2018 selloff and the March 2020 COVID crash

### Stationarity Testing
A stationary series keeps a stable average and spread over time. Most forecasting models learn better from stationary data, so the price series was tested four ways before modeling.

| Test | Finding |
|---|---|
| Histogram | Values spread unevenly across levels, not centered on one value |
| Seasonal decomposition, additive and multiplicative | A strong upward trend with small seasonal effects |
| Autocorrelation (ACF) | Correlation stayed near 1.0 across more than 30 lags |
| Partial autocorrelation (PACF) | Only the first lag mattered |
| Augmented Dickey Fuller test | p value of 0.84, so the series is **not stationary** |

The ACF and PACF together say something important. Once yesterday's price is known, older prices add almost no information.

### Approach
1. Scaled the closing price series
2. Built sliding windows where each sample uses the previous 10 days to predict the next day
3. Trained an LSTM network on data through late 2019
4. Validated on late 2019 through November 2020, then tuned hyperparameters across three comparison runs
5. Tested the final model on a 4 day holdout

### Results

| Date | Actual | Predicted | Error |
|---|---|---|---|
| Nov 17, 2020 | 3,623.11 | 3,615.15 | 0.22% |
| Nov 18, 2020 | 3,619.09 | 3,621.44 | 0.06% |
| Nov 19, 2020 | 3,585.22 | 3,624.92 | 1.11% |
| Nov 20, 2020 | 3,581.23 | 3,629.47 | 1.35% |

### What the Result Means
The errors look small, but the validation charts reveal the real pattern. The prediction line closely follows the actual line shifted by about one day. The model mostly learned that tomorrow's price will be close to today's price.

That is a reasonable guess, and the PACF confirmed it, but it has no real forecasting value. Anyone can predict that tomorrow will look like today without a neural network. This simple rule is called a **naive persistence baseline**, and because daily index moves are usually under 1%, it always looks accurate.

The holdout confirms the problem. The model predicted the index would rise from 3,615 to 3,629, while the index actually fell from 3,623 to 3,581. It got the direction wrong, which is what matters most for any trading or investment decision.

---

## Key Takeaways

1. **High scores can hide a weak model.** The inflation model scored 0.99 because its inputs contained the answer.
2. **Nonlinear models win when relationships have thresholds.** Random Forests captured income patterns that a straight line could not.
3. **Price level forecasts need a baseline.** Without comparing against "tomorrow equals today," a low error on stock prices proves very little.
4. **Stationarity testing is not optional.** The ADF, ACF, and PACF results explained the LSTM's behavior before it was even trained.

---

## Limitations and Next Steps

| Limitation | Next step |
|---|---|
| Inflation inputs are components of the target | Forecast next month's index from its own lagged values plus outside drivers such as interest rates, oil prices, and the exchange rate |
| GDP data is a single cross section | Use panel data across many years to forecast actual GDP growth |
| GDP Random Forest result relies on one split | Apply k fold cross validation to confirm the score |
| LSTM was trained on price levels | Model daily returns instead, which are close to stationary |
| No baseline comparison for the LSTM | Benchmark against a naive persistence model and report directional accuracy |
| Single train and test split across tracks | Use time series cross validation with rolling windows |

---

## Repository Structure

```
├── data/
│   ├── All_India_Index.csv
│   ├── countries_of_the_world.csv
│   └── sp500_2016_2020.csv
├── notebooks/
│   ├── 01_inflation_india_cpi.ipynb
│   ├── 02_gdp_per_capita.ipynb
│   └── 03_sp500_lstm.ipynb
├── report/
│   └── final_report.pdf
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/dhairya1310/Forecasting-Micro-and-Macro-Economic-Indicators-with-ML.git
cd Forecasting-Micro-and-Macro-Economic-Indicators-with-ML
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in order from 01 to 03.

---

## Tools and Technologies

| Category | Tools |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Deep learning | TensorFlow, Keras |
| Statistical testing | Statsmodels |
| Visualization | Matplotlib, Seaborn |

---

## Publication

Forecasting Macroeconomic Indicators with ML, presented at the Hinweis International Conference ASIT 2024.

---

## Author

**Dhairya Shah**
MS Data Science, Stevens Institute of Technology
[Portfolio](https://www.dhairya-shah.com) · [LinkedIn](https://www.linkedin.com/in/dhairya-shah-b00b111b6) · [GitHub](https://github.com/dhairya1310)
