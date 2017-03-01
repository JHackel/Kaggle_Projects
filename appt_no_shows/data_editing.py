import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('No-show-Issue-Comma-300k.csv')

# print(df.isnull().any()) # No nulls

# Age clean-up
df.loc[df.Age < 0, 'Age'] = df.loc[(df.Age >= 0) & (df.Age < 20), 'Age'].mean()
df.loc[df.Age >= 95, 'Age'] = df.loc[(df.Age < 95) & (df.Age > 50), 'Age'].mean()
# with sns.plotting_context("notebook", font_scale=1.5):
#     sns.set_style("whitegrid")
#     sns.distplot(df["Age"],
#                  bins=80,
#                  kde=False,
#                  color="tomato")
#     sns.plt.title("Age Distribution")
#     plt.ylabel("Count")
#     plt.show()

# AwaitingTime clean-up
df.loc[df.AwaitingTime <= -100.0, 'AwaitingTime'] = df.loc[(df.AwaitingTime >= -100.0) & (df.AwaitingTime <= -60),'AwaitingTime'].mean()
# with sns.plotting_context("notebook", font_scale=1.5):
#      sns.set_style("whitegrid")
#      sns.distplot(df["AwaitingTime"],
#                   bins=80,
#                   kde=False,
#                   color="tomato")
#      sns.plt.title("Waiting Time Distribution")
#      plt.ylabel("Count")
#      plt.show()

# Status clean-up
df['Status'] = df['Status'].map({'No-Show':0,'Show-Up':1})


# print(df.describe())

#throw_away = pd.DataFrame(df,columns=['Gender','AppointmentRegistration','ApointmentData','DayOfTheWeek'])

# Drop possibly unnecessary columns
df = df.drop(['Gender','AppointmentRegistration','ApointmentData','DayOfTheWeek'], axis=1)

#Scale values between 0 and 1
scaler = MinMaxScaler()
df[list(df)] = scaler.fit_transform(df[list(df)])



df.to_csv('No-show-edited.csv', index=False)
















