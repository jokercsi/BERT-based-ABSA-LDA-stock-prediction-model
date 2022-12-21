Download the dataset from LAB's shared_dir
`file:\\133.2.208.51\shared_dir\Students\2022\B4\kim\実験\data\`
<br>
<br>
download data folder and replace it to `absa-lda-stock-prediction\data`

<br>
<br>

## Steps of Data-Processing
### 1. Price Data
get .csv file using by yfinance 
 - Dow 30
 - Nasdaq
 - S&P 500

[Process of extracting price data](data-processing/priceTimeSeries.ipynb)

### 2. Text Data
orginize Date format in order to use stock prediction
 - CNBC
 [Process of Data-Processing](data-processing/cnbc_headlines.ipynb)
 - The Guardians
 [Process of Data-Processing](data-processing/guardian_headlines.ipynb)
 - Reuters
 [Process of Data-Processing](data-processing/reuters_headlines.ipynb)