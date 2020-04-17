# predicting prices for auction based on previous year data
x = players_r_D.iloc[:, [2, 3, 4, 5]].values
y = players_r_D.iloc[:, -1].values

# cleaning the data (converting to int)
for i in range(len(y)):
    num = ""
    for j in range(len(y[i])):
        if y[i][j] != "," and y[i][j] != "$":
            num = num + y[i][j]
    y[i] = float(num)
# taking care of the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])], remainder="passthrough")
x = ct.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)

from sklearn.tree import DecisionTreeRegressor
#fitting the regressor
reg = DecisionTreeRegressor()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
accuracy = len(y_test)
for i in range(len(y_test)):
    accuracy -= abs((abs(y_test[i] - y_pred[i]) / y_test[i]))
print("\nmodel has accuracy of {}".format(accuracy * 100 / len(y_test)))