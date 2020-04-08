import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here
tennis_stats = pd.read_csv("tennis_stats.csv")
df = pd.DataFrame(tennis_stats)
print(df.head)


# perform exploratory analysis here:
y = df[['Winnings']]
x = df[['ServiceGamesPlayed', 'Wins', 'BreakPointsFaced', 'ServiceGamesWon', 'BreakPointsSaved', 'FirstServeReturnPointsWon', 'BreakPointsConverted', 'ReturnGamesWon', 'TotalPointsWon', 'Aces']]


#plt.scatter(x,y)
#plt.show()

lrm = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

lrm.fit(x_train,y_train)

y_predict = lrm.predict(x_test)

print(lrm.score(x_train,y_train))

print(lrm.score(x_test, y_test))

plt.scatter(y_test, y_predict, alpha = 0.4)
plt.show()



