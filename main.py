import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

gdp_data = pd.read_excel('DELTA_GDP_NOR.xlsx')
rd_data = pd.read_excel('R&D_NOR.xlsx')
gdp_data['year'] = pd.to_datetime(gdp_data['year'])
rd_data['year'] = pd.to_datetime(rd_data['year'])
# Объединение данных по дате
data = pd.merge(gdp_data, rd_data, on='year')

# Предварительная обработка данных
data['year'] = pd.to_datetime(data['year'])
data.set_index('year', inplace=True)


# Функция для проведения теста Дики-Фуллера
def adfuller_test(series, signif=0.05):
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    return p_value < signif


# Проверка переменных на стационарность
is_gdp_stationary = adfuller_test(data['gdp'])
is_rd_stationary = adfuller_test(data['rd_spendings'])

print(f"GDP is Stationary: {is_gdp_stationary}")
print(f"R&D Spendings are Stationary: {is_rd_stationary}")

# Дифференцирование
data['diff'] = data['rd_spendings'].diff()

# Логарифмическое преобразование
data['log'] = np.log(data['rd_spendings'])

# Преобразование Бокса-Кокса
data['boxcox'], _ = boxcox(data['rd_spendings'])

# Проверка каждой версии временного ряда на стационарность
print(adfuller_test(data['diff'].dropna()),
      adfuller_test(data['log'].dropna()),
      adfuller_test(data['boxcox'].dropna()))

# Так как данные стационарны, можно приступить к оценке модели ARDL
# Например, модель ARDL(3,3) для наших данных
model = sm.tsa.arima.ARIMA(data['gdp'], order=(3, 3, 3), exog=data['rd_spendings'])
model_fit = model.fit()

# Вывод результатов
print(model_fit.summary())

# Визуализация результатов
data['model_predictions'] = model_fit.fittedvalues
plt.figure(figsize=(13, 7))
plt.plot(data['gdp'], label='GDP growth rate')
plt.plot(data['model_predictions'], label='Model Predictions', color='red')
plt.title('GDP growth rate and ARDL Model Predictions')
plt.legend()
plt.show()