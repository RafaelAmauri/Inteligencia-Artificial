import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

df = pd.read_csv("FuelConsumptionCo2.csv")

'''
LETRA A
'''

'''
for column in df.columns[0:-1]:
    print(f"Showing chart for {column}")
    plt.scatter(df[column], df["CO2EMISSIONS"])
    plt.show()
'''


'''
LETRA B
'''

'''
corr_matrix = df.corr().abs().sort_values(by=['CO2EMISSIONS'], ascending=False)
print(corr_matrix)
'''

'''
Variável mais correlacionada:

FUELCONSUMPTION_COMB_MPG
'''

'''
# 80% Pra treino e 20% pra teste
PERCENTAGE_TRAIN_DATA = 0.8

train = df["FUELCONSUMPTION_COMB_MPG"][ : int(PERCENTAGE_TRAIN_DATA*len(df["FUELCONSUMPTION_COMB_MPG"]))]
test  = df["FUELCONSUMPTION_COMB_MPG"][int(PERCENTAGE_TRAIN_DATA*len(df["FUELCONSUMPTION_COMB_MPG"])) + 1 :]
'''