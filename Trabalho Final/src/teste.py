import pandas as pd
'''
# Lista de paises
Paises = ["Brazil"]

# Lendo o CSV
csv_report = pd.read_csv('../assets/data.csv')
csv_report = csv_report[csv_report['Country Name'].isin(Paises)]

print(csv_report)
csv_report.to_csv('../assets/BR.csv', index=False)

csv_report = pd.read_csv('../assets/BR.csv')
csv_report = csv_report[csv_report['Country Name'].isin(Paises)]

print(csv_report)
'''