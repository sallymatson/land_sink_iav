import matplotlib.pyplot as plt


def monthly_plots():
	fig, ax = plt.subplots(1, 3, figsize=(20, 5))

	sns.lineplot(x=df2['Month'], y=df2['land_sink_North'], hue=df2['Year'], ax=ax[0])
	ax[0].set_title("Land sink in the Northern exatropics")

	sns.lineplot(x=df2['Month'], y=df2['land_sink_Tropics'], hue=df2['Year'], ax=ax[1])
	ax[1].set_title("Land sink in the Tropics")

	sns.lineplot(x=df2['Month'], y=df2['land_sink_South'], hue=df2['Year'], ax=ax[2])
	ax[2].set_title("Land sink in the Southern exatropics")


	plt.show()


def reg_results(df, reg, X, y, dependent_variable):
	plt.figure(figsize=(20,10))
	plt.plot(df['Date'], reg.predict(X), label="Predicted")
	plt.plot(df['Date'], y, label="Actual")
	plt.legend()
	plt.title(f"Predicted and actual {dependent_variable} over time")