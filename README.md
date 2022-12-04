<div id="top"></div>

# BERT based Aspect-Based Sentiment Analysis with LDA topic model for stock price prediction

観点感情分析結果を利用したトピックモデルによる株価推定

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
      <ul>
        <li><a href="#development-enviroment">Development Enviroment</a></li>
        <li><a href="#data">Data</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>
</br>
</br>

## Overview

stock price prediction
&nbsp;
</br>
</br>

### Development Enviroment

> #### Software
>
> - Windows 10
> - Python 3.10.8
> - pip3

> #### Library
>
> - transformers
> - yfinance
> - pandas
> - numpy
> - matplotlib
>   &nbsp;

### Data

- Text Data : [Headlines related to U.S. businesses (Kaggle)](https://www.kaggle.com/datasets/notlucasp/financial-news-headlines)
  1. CNBC
  2. The Guardian
  3. Reuters
- Stock Price Data : [3 Major U.S. Indexes (Yahoo! Finance)](https://finance.yahoo.com/)

  1. S&P 500 (^GSPC)
  2. NASDAQ Composite (^IXIC)
  3. Dow Jones Industrial Average (^DJI)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

_You need to install all._

- npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app._

bert를 사용하기 위한 tensorflow 버전
https://github.com/google-research/bert/issues/1140

1. Clone the repo
   ```sh
   git clone http://133.2.208.93/kim/absa-lda-stock-prediction.git
   ```
2. Install packages
   ```sh
   pip3 install -r requirements.txt
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space.

_For more examples_

<p align="right">(<a href="#top">back to top</a>)</p>
