#importing libraries

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

#Importing the Dataset

df = pd.read_csv('SBAnational.csv')
df_copy = df.copy()

#Data Pre-Processing

print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

df.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist','RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)
print(df.isnull().sum())
print(df.dtypes)
print(df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].head())
df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = \
print(df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',', '')))
print(df['ApprovalFY'].apply(type).value_counts())

print(df.ApprovalFY.unique())
def clean_str(x):
    if isinstance(x, str):
        return x.replace('A', '')
    return x

print(df['ApprovalFY'].apply(clean_str).astype('int64'))
print(df['ApprovalFY'].apply(type).value_counts())
df = df.astype({'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float', 'BalanceGross': 'float',
                          'ChgOffPrinGr': 'float', 'GrAppv': 'float', 'SBA_Appv': 'float'})
print(df.dtypes)
df['Industry'] = df['NAICS'].astype('str').apply(lambda x: x[:2])
df['Industry'] = df['Industry'].map({
    '11': 'Ag/For/Fish/Hunt',
    '21': 'Min/Quar/Oil_Gas_ext',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale_trade',
    '44': 'Retail_trade',
    '45': 'Retail_trade',
    '48': 'Trans/Ware',
    '49': 'Trans/Ware',
    '51': 'Information',
    '52': 'Finance/Insurance',
    '53': 'RE/Rental/Lease',
    '54': 'Prof/Science/Tech',
    '55': 'Mgmt_comp',
    '56': 'Admin_sup/Waste_Mgmt_Rem',
    '61': 'Educational',
    '62': 'Healthcare/Social_assist',
    '71': 'Arts/Entertain/Rec',
    '72': 'Accom/Food_serv',
    '81': 'Other_no_pub',
    '92': 'Public_Admin'
})
df.dropna(subset = ['Industry'], inplace = True)

print(df.FranchiseCode.unique())
df.loc[(df['FranchiseCode'] <= 1), 'IsFranchise'] = 0
df.loc[(df['FranchiseCode'] > 1), 'IsFranchise'] = 1
print(df.FranchiseCode)

df = df[(df['NewExist'] == 1) | (df['NewExist'] == 2)]

df.loc[(df['NewExist'] == 1), 'NewBusiness'] = 0
df.loc[(df['NewExist'] == 2), 'NewBusiness'] = 1
print(df.NewExist.unique())

print(df.RevLineCr.unique())

print(df.LowDoc.unique())

df = df[(df.RevLineCr == 'Y') | (df.RevLineCr == 'N')]
df = df[(df.LowDoc == 'Y') | (df.LowDoc == 'N')]

df['RevLineCr'] = np.where(df['RevLineCr'] == 'N', 0, 1)
df['LowDoc'] = np.where(df['LowDoc'] == 'N', 0, 1)
print(df.RevLineCr.unique())
print(df.LowDoc.unique())

print(df.MIS_Status.unique())
print(df.MIS_Status.value_counts())

df['Default'] = np.where(df['MIS_Status'] == 'P I F', 0, 1)
print(df['Default'].value_counts())

df[['ApprovalDate', 'DisbursementDate']] = df[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)
df['DaysToDisbursement'] = df['DisbursementDate'] - df['ApprovalDate']
print(df.DaysToDisbursement.info())

df['DaysToDisbursement'] = df['DaysToDisbursement'].astype('str').apply(lambda x: x[:x.index('d') - 1]).astype('int64')
df['DisbursementFY'] = df['DisbursementDate'].map(lambda x: x.year)
df['StateSame'] = np.where(df['State'] == df['BankState'], 1, 0)
df['SBA_AppvPct'] = df['SBA_Appv'] / df['GrAppv']
df['AppvDisbursed'] = np.where(df['DisbursementGross'] == df['GrAppv'], 1, 0)
print(df.dtypes)

df = df.astype({'IsFranchise': 'int64', 'NewBusiness': 'int64'})
print(df.dtypes)

df.drop(columns=['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode',
                      'ChgOffDate', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr', 'SBA_Appv', 'MIS_Status'], inplace=True)
print(df.isnull().sum())

print(df.shape)

print(len(df.Term.unique()))


df['RealEstate'] = np.where(df['Term'] >= 240, 1, 0)
df['GreatRecession'] = np.where(((2007 <= df['DisbursementFY']) & (df['DisbursementFY'] <= 2009)) |
                                     ((df['DisbursementFY'] < 2007) & (df['DisbursementFY'] + (df['Term']/12) >= 2007)), 1, 0)
print(df.DisbursementFY.unique())

df = df[df.DisbursementFY <= 2010]
print(df.shape)

print(df.describe(include = ['int', 'float', 'object']))
df['DisbursedGreaterAppv'] = np.where(df['DisbursementGross'] > df['GrAppv'], 1, 0)
print(df.DisbursedGreaterAppv.unique())
df = df[df['DaysToDisbursement'] >= 0]

print(df.shape)
print(df.describe(include = ['int', 'float', 'object']))

#data visualization:-

correlation_figure, correlation_axis = plt.subplots(figsize = (30,25))
corr_mtrx = df.corr()
correlation_axis = sns.heatmap(corr_mtrx, annot= True)

plt.xticks(rotation = 30, horizontalalignment = 'right', fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()

industry_group = df.groupby(['Industry'])
df_industrySum = industry_group.sum().sort_values('DisbursementGross', ascending = False)
df_industryAve = industry_group.mean().sort_values('DisbursementGross', ascending=False)

fig = plt.figure(figsize=(40,20))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.bar(df_industrySum.index, df_industrySum['DisbursementGross'] / 1000000000)
ax1.set_xticklabels(df_industrySum.index, rotation=30, horizontalalignment='right', fontsize=10)

ax1.set_title('Gross SBA Loan Disbursement by Industry from 1984-2010', fontsize=20)
ax1.set_xlabel('Industry', fontsize = 20)
ax1.set_ylabel('Gross Loan Disbursement (Billions)', fontsize = 20)

ax2.bar(df_industryAve.index, df_industryAve['DisbursementGross'])
ax2.set_xticklabels(df_industryAve.index, rotation=30, horizontalalignment='right', fontsize=10)

ax2.set_title('Average SBA Loan Disbursement by Industry from 1984-2010', fontsize=20)
ax2.set_xlabel('Industry',  fontsize = 20)
ax2.set_ylabel('Average Loan Disbursement',  fontsize = 20)

plt.show()

fig2, ax = plt.subplots(figsize = (30,15))

ax.bar(df_industryAve.index, df_industryAve['DaysToDisbursement'].sort_values(ascending=False))
ax.set_xticklabels(df_industryAve['DaysToDisbursement'].sort_values(ascending=False).index, rotation=35,
                   horizontalalignment='right', fontsize=10)

ax.set_title('Average Days to SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
ax.set_xlabel('Industry')
ax.set_ylabel('Average Days to Disbursement')

plt.show()

fig3 = plt.figure(figsize=(50, 30))

ax1a = plt.subplot(2,1,1)
ax2a = plt.subplot(2,1,2)

def stacked_setup(df, col, axes, stack_col= 'Default'):
    data = df.groupby([col, stack_col])[col].count().unstack(stack_col)
    data.fillna(0)

    axes.bar(data.index, data[1], label='Default')
    axes.bar(data.index, data[0], bottom=data[1], label='Paid in full')

stacked_setup(df=df, col='Industry', axes=ax1a)
ax1a.set_xticklabels(df.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default').index,
                     rotation=35, horizontalalignment='right', fontsize=10)

ax1a.set_title('Number of PIF/Defaulted Loans by Industry from 1984-2010', fontsize=10)
ax1a.set_xlabel('Industry')
ax1a.set_ylabel('Number of PIF/Defaulted Loans')
ax1a.legend()

# Number of Paid in full and defaulted loans by State
stacked_setup(df=df, col='State', axes=ax2a)

ax2a.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize= 20)
ax2a.set_xlabel('State')
ax2a.set_ylabel('Number of PIF/Defaulted Loans')
ax2a.legend()

plt.tight_layout()
plt.show()

def_ind = df.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default')
def_ind['Def_Percent'] = def_ind[1]/(def_ind[1] + def_ind[0])

print(def_ind)

def_state = df.groupby(['State', 'Default'])['State'].count().unstack('Default')
def_state['Def_Percent'] = def_state[1]/(def_state[1] + def_state[0])

print(def_state)

fig4, ax4 = plt.subplots(figsize = (30,15))

stack_data = df.groupby(['DisbursementFY', 'Default'])['DisbursementFY'].count().unstack('Default')

x = stack_data.index
y = [stack_data[1], stack_data[0]]

ax4.stackplot(x, y, labels = ['Default', 'Paid In Full'])
ax4.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize = 30)

ax4.set_xlabel('Disbursement Year')
ax4.set_ylabel('Number of PIF/Defaulted Loans')
ax4.legend(loc='upper left', fontsize = 20)

plt.show()
print(df)

#data training and testing

df = pd.get_dummies(df)
df.head()

y = df['Default']
X = df.drop('Default', axis = 1)
scale = StandardScaler()
X_scld = scale.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scld, y, test_size=0.25)

#MODEL-1) LogisticRegression
#------------------------------------------
clf= HistGradientBoostingClassifier()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-1: Accuracy of LogisticRegression : ", acc_logreg  )

#MODEL-2) Gaussian Naive Bayes
#------------------------------------------
from sklearn.naive_bayes import GaussianNB
clf= HistGradientBoostingClassifier()
gaussian = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-2: Accuracy of GaussianNB : ", acc_gaussian  )

#MODEL-3) Support Vector Machines
#------------------------------------------
clf= HistGradientBoostingClassifier()
svc = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-3: Accuracy of Support Vector Machines : ", acc_svc  )

#MODEL-4) Decision Tree Classifier
#------------------------------------------
from sklearn.tree import DecisionTreeClassifier
clf= HistGradientBoostingClassifier()
decisiontree = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-4: Accuracy of DecisionTreeClassifier : ", acc_decisiontree  )

#MODEL-5) Random Forest
#------------------------------------------
from sklearn.ensemble import RandomForestClassifier
clf= HistGradientBoostingClassifier()
randomforest = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-5: Accuracy of RandomForestClassifier : ",acc_randomforest  )

#MODEL-6) KNN or k-Nearest Neighbors
#------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
clf= HistGradientBoostingClassifier()
knn = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-6: Accuracy of k-Nearest Neighbors : ",acc_knn  )

#Let's compare the accuracies of each model!

models = pd.DataFrame({
                        'Model': ['Logistic Regression','Gaussian Naive Bayes','Support Vector Machines',
                                   'Decision Tree',     'Random Forest',       'KNN',],
                        'Score': [acc_logreg, acc_gaussian, acc_svc,
                                  acc_decisiontree, acc_randomforest,  acc_knn]
                    })

print()
print( models.sort_values(by='Score', ascending=False) )
print()

#print the confusion_matrix for prediction accuracy of LogicsticRegression
matrix = confusion_matrix(y_val,y_pred)
print("confusion_matrix for prediction accuracy of LogicsticRegression = \n")
print(matrix)
print("\n\n")

#Vizualization: heatmap of confusion_matrix for LogicsticRegression
sns.heatmap(confusion_matrix(y_val,y_pred), annot=True, cmap='Blues')
plt.show()

#Train the model-03: SupportVectorClassifier Algoritmns
clf= HistGradientBoostingClassifier()
model=SVC()
clf.fit(X_train,y_train)

ax=sns.distplot(y_val,hist=False,label='Actual Values')
ax=sns.distplot(y_pred,hist=False,label='Predicted Values')
ax.set_title('SupportVectorClassifier')
plt.legend()
plt.show()

#print the confusion_matrix for prediction accuracy of SupportVectorClassifier
matrix = confusion_matrix(y_val,y_pred)
print("confusion_matrix for prediction accuracy of SupportVectorClassifier = \n")
print(matrix)
print("\n\n")


from sklearn.pipeline import make_pipeline
clf = SVC()  # Choose your desired classifier
pipeline_model = make_pipeline(StandardScaler(), clf)
# Model Training and Evaluation
clf= HistGradientBoostingClassifier()
clf.fit(X_train, y_train)

# Make Predictions on the validation set
y_pred = clf.predict(X_val)

# Evaluate the model performance
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))
print("Confusion Matrix:\n", conf_matrix)

# Cross-Validation using Pipelining
k_fold = KFold(n_splits=10, random_state=7, shuffle=True)

results = cross_val_score(clf, X, y, cv=k_fold)
print("Accuracy from Cross-Validation: %.2f%%" % (results.mean() * 100))