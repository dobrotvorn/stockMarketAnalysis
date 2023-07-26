# ваш код модели на STAN и её обучения моделей здесь

import pystan  # probabilistic programming language
import arviz as az  # visualizations for bayesian approach
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error as MAE
from sktime.forecasting.naive import NaiveForecaster


def prophnet(train_df22, test_df22):
    m = Prophet(daily_seasonality=False)
    train_df22.columns = ['ds', 'y']
    test_df22.columns = ['ds', 'y']
    m.fit(train_df22)
    predicted_df_profnet = m.predict(test_df22)
    err_prophnet = MAE(predicted_df_profnet['yhat'].to_numpy(), test_df22['y'])
    print(err_prophnet)
    m.plot(predicted_df_profnet)
    test_df22.index = test_df22['ds']
    test_df22['y'].plot()
    plt.show()


def dlt(train_df11, test_df11):
    lgt = LGTFull(
        response_col='YNDX',
        date_col='date',
        seasonality=2,
        seed=8888,
    )
    lgt.fit(train_df11)
    y_pred_lgt = lgt.predict(test_df11)

    dlt = DLTFull(
        response_col='размерчик',
        date_col='дата',
        seasonality=12,
        seed=8888,
    )
    dlt.fit(train_df11)
    y_pred_dlt = dlt.predict(test_df11)
    plot_predicted_data(training_actual_df=train_df11, predicted_df=y_pred_dlt,
                        date_col='date', actual_col='YNDX',
                        test_actual_df=test_df11, title='Prediction with LGTFull Model')


def naiveSolver(data, ticker=None):
    train, test = data['train'], data['test']
    forecaster_naive = NaiveForecaster(strategy="drift")
    a = train[ticker]
    forecaster_naive.fit(train[ticker])
    y_pred_naiv = forecaster_naive.predict(fh=list(range(len(test[ticker]))))
    # y_pred_naiv.plot()
    # test_df2.plot()
    # second strategy
    forecaster_naive_season = NaiveForecaster(strategy="last", sp=12)
    forecaster_naive_season.fit(train[ticker])
    y_pred_naiv_season = forecaster_naive_season.predict(fh=list(range(len(test[ticker]))))
    y_pred_next_day = forecaster_naive_season.predict(fh=list(range(len(test[ticker]) + 1))).tail(1)
    y_pred_next_day_season = forecaster_naive.predict(fh=list(range(len(test[ticker]) + 1))).tail(1)
    #можно еще сделать распределение - несколько параметров по стратегиям, для самого лучшего мае
    #  сделать самое больше количестов повторений, а потом все в диаграмку засунуть
    return {'test_prediction': {'y_naiv': y_pred_naiv, 'y_naiv_season' :y_pred_naiv_season},
            'name': ['naive', 'naive_season'],
            'mae': [MAE(y_pred_naiv, test[ticker]), MAE(y_pred_naiv_season, test[ticker])],
            'nex_day_prediction': [y_pred_next_day, y_pred_next_day_season]}





def pystanModel(timeseries, ticker=None):
    model_code = """
    data {
      int<lower=0> n;         // number of observations
      vector[n] y;            // time series data
      vector[n] x;            // predictor
    }
    parameters {
      // equation parameters
      real<lower=0, upper=1> alpha;   // alpha
      real<lower=0, upper=1> beta;    // beta
      real<lower=0, upper=1> gamma;    // gamma
      real<lower=0, upper=1> theta;    // theta
      real k;                          // k
      real<lower=0> sigma;             // sigma
    
      // initial values
      real linit;        // linit
      real binit;        // binit
      vector[12] sinit;  // sinit
    }
    transformed parameters {
      vector[n+1] l;  // l
      vector[n+1] b;  // b
      vector[n+12] s;  // s
      vector[n] r;     // r
      vector[n] yhat;  // E(y_t | F_{t-1})
    
      // initial observations
      l[1] = linit;      // l
      b[1]=  binit;      // b
      for (t in 1:12) { // s
        s[t] = sinit[t];
      }          
    
      // update equations
      for (t in 1:n) {
    
        int tp; 
        int ts;
    
        tp = t + 1;
        ts = t + 12;
    
        r[t] = k * x[t];
        b[tp] = b[tp-1] * (1 - beta) + beta * (y[t] - l[tp-1] - s[ts-12]);
        s[ts] = (1- gamma) * s[ts-12] + gamma * (y[t] - l[tp-1] - b[tp-1]);
        l[tp] = (1 - alpha) * (l[tp-1] + b[tp-1])  + alpha * (y[t] - s[ts-12]);
        yhat[t] = l[tp-1] + b[tp-1] + s[ts-12];
      }
    }
    
    model {
      // prior for e-parameters
      alpha ~ uniform(0, 1);
      beta ~ uniform(0, 1);
      gamma ~ uniform(0, 1);
      sigma ~ normal(0, 10) T[0, ];  // T[a, b] or T[a, ] or T[, b] means truncated
      k ~ normal(0, 10);
    
      // prior for initial values
      linit ~ normal(0, 10);
      binit ~ normal(0, 10);
      sinit ~ normal(0, 10);   
    
    
      // likelihood for observed data
      for (t in 1:n) {
        y[t] ~ normal(yhat[t], sigma);
      }
    }
    """

    model = pystan.StanModel(model_code=model_code)
    timeseries = timeseries.to_list()
    length = len(timeseries)
    x = np.ones(length)
    air_data = {'n': length, 'y': timeseries.values, 'x': x}
    post = model.sampling(air_data, chains=2, iter=5000, warmup=1000)
    train_df1_81_hat = post['l[81]']+ post['theta'] * post['b[81]'] + post['k'] * 1 + np.random.normal(loc=0, scale=post['sigma'])
    print(np.median(train_df1_81_hat)) # точечный прогноз на сентябрь 2019 ( 2019-09)
    return train_df1_81_hat