import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#########WEEK 5#########

########EXERCISE 2##########
#########FAIRNESS-AWARE AI#########

#########SIMULATING THE DATA#########

#sample size n
n = 5000
#gender
gen = np.random.binomial(1, 0.5, size=n)
#work hours
hrs = np.random.binomial(60, 0.5, size=n)
#salary = 100 * hours + noise (std.deviation 10)
sal = hrs * np.random.normal(100, 10, size=n)
#create a dataframe
data = pd.DataFrame({'Gender': gen, 'Hours': hrs, 'Salary': sal})

#########PLOT THE SIMULATED DATA#########

#plot men, women (+ trend line total) on separate layers
sns.regplot(x='Hours',
            y='Salary',
            data=data[data['Gender']==1],
            color='darkBlue',
            scatter_kws={'alpha':0.5})
sns.regplot(x='Hours',
            y='Salary',
            data=data[data['Gender']==0],
            color='darkOrange',
            scatter_kws={'alpha':0.5})
sns.regplot(x='Hours',
            y='Salary',
            data=data,
            marker='None',
            color='red')
plt.show()

#calculate the slope and print it
reg = LinearRegression().fit(hrs.reshape(-1,1), sal.reshape(-1,1))
#print out the slope
print('slope: %.1f' % reg.coef_)

#########1 ALTERING THE DATA 1#########

#salary of women is reduced by 200€ + plot + slope
#copy the original data
#loop through data copy to reduce salary from women by 200€

data_alt1 = data.copy()
for index, row in data_alt1.iterrows():
    if row['Gender'] == 1:
        data_alt1.at[index, 'Salary'] = row['Salary'] - 200


#########1 PLOTTING THE ALTERED DATA 1#########

#plot men, women and (trend line total) on separate layers
sns.regplot(x='Hours',
            y='Salary',
            data=data_alt1[data_alt1['Gender']==1],
            color='darkBlue',
            scatter_kws={'alpha':0.5})
sns.regplot(x='Hours',
            y='Salary',
            data=data_alt1[data_alt1['Gender']==0],
            color='darkOrange',
            scatter_kws={'alpha':0.5})
sns.regplot(x='Hours',
            y='Salary',
            data=data_alt1,
            marker='None',
            color='red')
plt.show()

alt1_hours = data_alt1['Hours'].to_numpy()
alt1_salary = data_alt1['Salary'].to_numpy()
#calculate the slope and print it
reg = LinearRegression().fit(alt1_hours.reshape(-1,1), alt1_salary.reshape(-1,1))
#print out the slope
print('slope: %.1f' % reg.coef_)

#########2 ALTERING THE DATA 2#########

#working hours of men distributed (60, 0.55)
#working hours of women distributed (60, 0.45)

#copy the original data
#loop through data copy to redistribute working hours
data_alt2 = data.copy()
for index, row in data_alt2.iterrows():
    if row['Gender'] == 1:
        data_alt2.at[index, 'Hours'] = np.random.binomial(60, 0.45)
        data_alt2.at[index, 'Salary'] = row['Hours'] * np.random.normal(100, 10)
    else:
        data_alt2.at[index, 'Hours'] = np.random.binomial(60, 0.55)
        data_alt2.at[index, 'Salary'] = row['Hours'] * np.random.normal(100, 10)

#########2 PLOTTING THE ALTERED DATA 2#########

sns.regplot(x='Hours',
            y='Salary',
            data=data_alt2[data_alt2['Gender']==1],
            color='darkBlue',
            scatter_kws={'alpha':0.5})
sns.regplot(x='Hours',
            y='Salary',
            data=data_alt2[data_alt2['Gender']==0],
            color='darkOrange',
            scatter_kws={'alpha':0.5})
sns.regplot(x='Hours',
            y='Salary',
            data=data_alt2,
            marker='None',
            color='red')
plt.show()

alt2_hours = data_alt2['Hours'].to_numpy()
alt2_salary = data_alt2['Salary'].to_numpy()
#calculate the slope and print it
reg = LinearRegression().fit(alt2_hours.reshape(-1,1), alt2_salary.reshape(-1,1))
#print out the slope
print('slope: %.1f' % reg.coef_)

#########3 ALTERING THE DATA 3#########

#both factors above used at the same time

#copy the original data
#loop through data copy to redistribute working hours and
#reduce women's salary

data_alt3 = data.copy()
for index, row in data_alt3.iterrows():
    if row['Gender'] == 1:
        data_alt3.at[index, 'Hours'] = np.random.binomial(60, 0.45)
        data_alt3.at[index, 'Salary'] = row['Hours'] * np.random.normal(100, 10)
    else:
        data_alt3.at[index, 'Hours'] = np.random.binomial(60, 0.55)
        data_alt3.at[index, 'Salary'] = row['Hours'] * np.random.normal(100, 10)

for index, row in data_alt3.iterrows():
    if row['Gender'] == 1:
        data_alt3.at[index, 'Salary'] = row['Salary'] - 200

#########3 PLOTTING THE ALTERED DATA 3#########

sns.regplot(x="Hours",
            y="Salary",
            data=data_alt3[data_alt3["Gender"]==1],
            color="darkBlue",
            scatter_kws={'alpha':0.5})
sns.regplot(x="Hours",
            y="Salary",
            data=data_alt3[data_alt3["Gender"]==0],
            color="darkOrange",
            scatter_kws={'alpha':0.5})
sns.regplot(x="Hours",
            y="Salary",
            data=data_alt3,
            marker="None",
            color="red")
plt.show()

alt3_hours = data_alt3['Hours'].to_numpy()
alt3_salary = data_alt3['Salary'].to_numpy()
#calculate the slope and print it
reg = LinearRegression().fit(alt3_hours.reshape(-1,1), alt3_salary.reshape(-1,1))
#print out the slope
print('slope: %.1f' % reg.coef_)