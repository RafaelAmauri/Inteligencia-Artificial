import predictor

pred = predictor.Predictor("../assets/BR.csv", ['SP.POP.TOTL', 'SP.RUR.TOTL'])
pred.plot_indicators()