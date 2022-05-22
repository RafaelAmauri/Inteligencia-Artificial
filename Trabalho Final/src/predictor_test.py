import predictor

pred = predictor.TimeSeriesPredictor("../assets/BR.csv", ['SP.POP.TOTL', 'SP.RUR.TOTL'], 1960, 2004)

#pred.plot_indicators()

pred.predict_indicators()