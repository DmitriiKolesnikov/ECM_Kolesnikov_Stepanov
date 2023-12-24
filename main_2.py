import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
# Но сначала нужно убедиться, что данные приведены к формату datetime и отформатированы корректно
# Преобразование строк с датами в datetime и установка в качестве индекса
gdp_data = pd.read_excel('DELTA_GDP_NOR.xlsx')
rd_data = pd.read_excel('R&D_NOR.xlsx')
gdp_data['year'] = pd.to_datetime(gdp_data['year'])
rd_data['year'] = pd.to_datetime(rd_data['year'])
gdp_data.set_index('year', inplace=True)
rd_data.set_index('year', inplace=True)

# Для теста Грейнджера данные должны быть совмещены по временной шкале
combined_data = pd.concat([gdp_data, rd_data], axis=1)
print(combined_data.shape)
# Убедимся, что пропущенные данные обработаны (например, заполнены или удалены)
combined_data.dropna(inplace=True)

# Тест Грейнджера
max_lags = 4  # Максимальное количество лагов для проверки
granger_test_results = grangercausalitytests(combined_data, max_lags, verbose=True)

# Вывод результатов теста
granger_test_results