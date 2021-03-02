import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Faz a leitura do arquivo CSV usando pandas
dataset = pd.read_csv ('pib.csv')

# Cria uma base contendo as variaveis independentes e uma base contendo a variavel dependente.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Separa a base em duas partes: uma para treinamento e outra para testes. Use 85% das instâncias para o treinamento.
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split (x, y, test_size=0.15, random_state=0)

# Constrói um modelo de regressão linear simples em função dos dados de treinamento.
linearRegression = LinearRegression()
linearRegression.fit(x_treinamento, y_treinamento)

# y_pred = linearRegression.fit (x_treinamento, y_treinamento)


# Exibe a reta obtida e os dados de treinamento em um mesmo grafico. Inclua a equacao obtida no título do grafco.
plt.scatter (x_treinamento, y_treinamento, color="green") 
plt.plot (x_treinamento, linearRegression.predict(x_treinamento), color="yellow") 
plt.title("PIB - Produto Interno Bruto Brasileiro - Per capita (treinamento)")
plt.xlabel ("Ano")
plt.ylabel ("U$ (Dolares)")
plt.show()

# Exibe a reta obtida e os dados de teste em um mesmo grafico. Inclua a equacao obtida no título do grafco.
plt.scatter (x_teste, y_teste, color="green") 
plt.plot (x_treinamento, linearRegression.predict(x_treinamento), color="yellow") 
plt.title("PIB - Produto Interno Bruto Brasileiro - Per capita (teste)")
plt.xlabel ("Ano")
plt.ylabel ("U$ (Dolares)")
plt.show()

# coeficiente
# print (f"{linearRegression.coef_[0]:.2f}x + {linearRegression.intercept_:2f}")