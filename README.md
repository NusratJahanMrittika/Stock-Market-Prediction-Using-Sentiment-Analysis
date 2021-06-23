# Stock-Market-Prediction-Using-Sentiment-Analysis
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Read and store data into variables
df1 = pd.read_csv(r'E:\533 ML\STOCK\STOCK\DATA\DJIA_News.csv')
df2 = pd.read_csv(r'E:\533 ML\STOCK\STOCK\DATA\DJIA_Stock.csv')

#Merge the data set on the date field
merge = df1.merge(df2, how='inner', on='Date', left_index = True)

#Combine the top news headlines
headlines = []
for row in range(0,len(merge.index)):
    headlines.append(' '.join(str(x) for x in merge.iloc[row,2:27]))
    
#Clean the data
clean_headlines = []
for i in range(0, len(headlines)):
  clean_headlines.append(re.sub("b[(')]+", '', headlines[i] ))
  clean_headlines[i] = re.sub('b[(")]+', '', clean_headlines[i] )
  clean_headlines[i] = re.sub("\'", '', clean_headlines[i] )
  
#Add the clean headlines to the data set
merge['Combined_News'] = clean_headlines

# Create a function to get the subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity
# Create a function to get the polarity
def getPolarity(text):
  return  TextBlob(text).sentiment.polarity

# Create two new columns 'Subjectivity' & 'Polarity'
merge['Subjectivity'] =merge['Combined_News'].apply(getSubjectivity)
merge['Polarity'] =merge['Combined_News'].apply(getPolarity)

#Create a function to get the sentiment scores (using Sentiment Intensity Analyzer)
def getSIA(text):
  sia = SentimentIntensityAnalyzer()
  sentiment = sia.polarity_scores(text)
  return sentiment

#Get the sentiment scores for each day
compound = []
neg = []
neu = []
pos = []
SIA = 0
for i in range(0, len(merge['Combined_News'])):
  SIA = getSIA(merge['Combined_News'][i])
  compound.append(SIA['compound'])
  neg.append(SIA['neg'])
  neu.append(SIA['neu'])
  pos.append(SIA['pos'])
  
#Store the sentiment scores in the data frame
merge['Compound'] =compound
merge['Negative'] =neg
merge['Neutral'] =neu
merge['Positive'] = pos

#Create a list of columns for the final output
columns_1 = [ 'Date', 'Open',  'High', 'Low',    'Volume', 'Subjectivity', 'Polarity', 'Compound',
                'Negative','Neutral' ,'Positive',  'Label' ]
df1 = merge[columns_1]


#Set display to show full output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Show the first 5 rows of the OUTPUT
print(df1.head(5))
print('\n\n')

#Convert the output list to a csv file
dfx = pd.DataFrame(df1)
dfx.to_csv('stock_file.csv')

#Create a list of columns with 'Date' column for training the data
columns_2 = [ 'Open',  'High', 'Low',    'Volume', 'Subjectivity', 'Polarity', 'Compound',
                'Negative','Neutral' ,'Positive',  'Label' ]
df2 = merge[columns_2]

#Create the feature data set
X = df2
X = np.array(X.drop(['Label'],1))

#Create the target data set
y = np.array(df2['Label'])

#Split the data into 70% training and 30% testing data sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

model = LinearDiscriminantAnalysis().fit(x_train, y_train)

#Get the models predictions/classification
predictions = model.predict(x_test)
print(predictions)
print('\n\n')

#Show the models metrics
print(classification_report(y_test, predictions))


