from sklearn import linear_model

def basic_linear_model(X, y):
	reg = linear_model.LinearRegression()
	reg.fit(X, y)
	return reg

