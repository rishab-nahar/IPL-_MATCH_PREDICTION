# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# importing the datasets
matches_D = pd.read_csv("matches.csv ")
players_r_D = pd.read_csv("player_rank.csv")
team_r_D = pd.read_csv("team_rank.csv")
winners_D = pd.read_csv("win.csv")

# making np arrays of data sets
matches = matches_D.iloc[:635, :-1].values

# creating dictionaries
city_name = {"Hyderabad": 1, "Pune": 2, "Rajkot": 3, "Indore": 4, "Bangalore": 5, "Mumbai": 6, "Kolkata": 7, "Delhi": 8,
             "Chandigarh": 9, "Kanpur": 10, "Jaipur": 11, "Chennai": 12, "Cape Town": 13, "Port Elizabeth": 14,
             "Durban": 15, "Centurion": 16, "East London": 17, "Johannesburg": 18, "Kimberley": 19, "Bloemfontein": 20,
             "Ahmedabad": 21, "Cuttack": 22, "Nagpur": 23, "Dharamshala": 24, "Kochi": 25, "Visakhapatnam": 26,
             "Raipur": 27, "Ranchi": 28, "Abu Dhabi": 29, "Sharjah": 30, "Dubai": 31, "Mohali": 32}
team_name = {"no result": 0, "Sunrisers Hyderabad": 1, "Mumbai Indians": 2, "Gujarat Lions": 3,
             "Rising Pune Supergiant": 4, "Royal Challengers Bangalore": 5, "Kolkata Knight Riders": 6,
             "Delhi Daredevils": 7, "Kings XI Punjab": 8, "Chennai Super Kings": 9, "Rajasthan Royals": 10,
             "Deccan Chargers": 11, "Kochi Tuskers Kerala": 12, "Pune Warriors": 13, "Rising Pune Supergiants": 14}
decision = {"field": 1, "bat": 2}
year = {2008: 1, 2009: 2, 2010: 3, 2011: 4, 2012: 5, 2013: 6, 2014: 7, 2015: 8, 2016: 9, 2017: 10, 2018: 11}
city_name = defaultdict(int, city_name)
team_name = defaultdict(int, team_name)
decision = defaultdict(lambda: 1, decision)

# cleaning the data(into numerical values)
teams = [0] * 15
for i in team_name:
    teams[team_name[i]] = i
tosswin = []
winners_dec = []
for i in matches:
    i[4] = team_name[i[4]]
    i[5] = team_name[i[5]]
    tosswin.append(team_name[i[6]])
    i[6] = team_name[i[6]]
    i[2] = city_name[i[2]]
    i[7] = decision[i[7]]
    winners_dec.append(team_name[i[10]])
    if team_name[i[10]] == i[4]:
        i[10] = 1
    else:
        i[10] = 0
    i[1] = year[i[1]]

# making dependant and independant variables
feature_set = matches[:, [1, 2, 4, 5, 6, 7]]
won = matches[:, 10].astype(int)
won_by_runs = matches[:, 11].astype(int)
sum_of_runs = 0
count = 0
for i in range(len(won_by_runs)):
    if won_by_runs[i] != 0:
        sum_of_runs += won_by_runs[i]
        count += 1
average = sum_of_runs // count
for i in range(len(won_by_runs)):
    if won_by_runs[i] == 0:
        won_by_runs[i] = average

won_by_wickets = matches[:, 12].astype(int)
sum_of_wickets = 0
count = 0
for i in range(len(won_by_wickets)):
    if won_by_wickets[i] != 0:
        sum_of_wickets += won_by_wickets[i]
        count += 1
average = sum_of_wickets // count
for i in range(len(won_by_wickets)):
    if won_by_wickets[i] == 0:
        won_by_wickets[i] = average

# prediction
print("Do you want to predict results...?(y/n)")
response = input()

# instructions
print(""" CITYNUMBERS:
      "Hyderabad": 1, "Pune": 2, "Rajkot": 3, "Indore": 4, "Bangalore": 5 
      "Mumbai": 6, "Kolkata": 7, "Delhi": 8, "Chandigarh": 9, "Kanpur": 10, 
      "Jaipur": 11, "Chennai": 12, "Cape Town": 13, "Port Elizabeth": 14, "Durban": 15,
      "Centurion": 16, "East London": 17, "Johannesburg": 18, "Kimberley": 19, "Bloemfontein": 20,
      "Ahmedabad": 21, "Cuttack": 22, "Nagpur": 23, "Dharamshala": 24, "Kochi": 25,
      "Visakhapatnam": 26, "Raipur": 27, "Ranchi": 28, "Abu Dhabi": 29, "Sharjah": 30, "Dubai": 31, "Mohali": 32 """)
print("""\nTEAMNUMBERS:
      "Sunrisers Hyderabad": 1, "Mumbai Indians": 2, "Gujarat Lions": 3, "Rising Pune Supergiant": 4,
      "Royal Challengers Bangalore": 5, "Kolkata Knight Riders": 6, "Delhi Daredevils": 7, "Kings XI Punjab": 8,
      "Chennai Super Kings": 9, "Rajasthan Royals": 10, "Deccan Chargers": 11, "Kochi Tuskers Kerala": 12,
      "Pune Warriors": 13, "Rising Pune Supergiants": 14""")
print("""\nTOSSWINNER
      1:team1  0:team2""")
print("""\nTOSSDECISION
      "field": 1, "bat": 2
      """)
while (response != "n" and response != "N"):
    print("""\nGive the following parameters based on following parameter seperated by a space
          year(2008-2017) ,city_number,team1_number,team2_number,tosswinner,toss_decision""")
    temp = [[int(i) for i in input().strip().split(" ")]]
    temp = np.array(temp)

    # training the model
    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier()
    classifier.fit(feature_set, won)
    from sklearn.svm import SVR

    regressor1 = SVR()
    regressor1.fit(feature_set, won_by_runs)
    regressor2 = SVR()
    regressor2.fit(feature_set, won_by_wickets)

    # predicting on input
    win = classifier.predict(temp)
    wick = 0
    run = 0
    if win == 1:
        if temp[0][4] == 1:
            if temp[0][5] == 1:
                wick = regressor2.predict(temp)
            else:
                run = regressor1.predict(temp)
        else:
            if temp[0][5] == 1:
                run = regressor1.predict(temp)
            else:
                wick = regressor2.predict(temp)
    else:
        if temp[0][4] == 1:
            if temp[0][5] == 1:
                run = regressor1.predict(temp)
            else:
                wick = regressor2.predict(temp)
        else:
            if temp[0][5] == 1:
                wick = regressor2.predict(temp)
            else:
                run = regressor1.predict(temp)

    # printing the results
    if win == 1:
        winner = teams[temp[0][2]]
        loser = teams[temp[0][3]]
        if run == 0:
            print("{} beats {} by {} wickets".format(winner, loser, wick[0]))
        else:
            print("{} beats {} by {} runs".format(winner, loser, run[0]))
    else:
        winner = teams[temp[0][3]]
        loser = teams[temp[0][2]]
        if run == 0:
            print("{} beats {} by {} wickets".format(winner, loser, wick[0]))
        else:
            print("{} beats {} by {} runs".format(winner, loser, run[0]))

    print("\nDo you want to predict results...?(y/n)")
    response = input()
