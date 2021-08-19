import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

human_data = pd.read_table('human_data.txt')
# h = human_data.head()
# print(h)
chimp_data = pd.read_table('chimp_data.txt')
dog_data = pd.read_table('dog_data.txt')
# chimp_data.head()
# dog_data.head()

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# def getKmers(sequence, size=6):
#     for x in range(len(sequence) - size + 1):
#         return sequence[x:x+size].lower()


human_data['Kmers'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
human_data = human_data.drop('sequence', axis=1)
chimp_data['Kmers'] = chimp_data.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp_data = chimp_data.drop('sequence', axis=1)
dog_data['Kmers'] = dog_data.apply(lambda x: getKmers(x['sequence']), axis=1)
dog_data = dog_data.drop('sequence', axis=1)

# b = human_data.head()
# print(b)

human_dlist = list(human_data['Kmers'])          # nested lists (many lists in 'human_dlist' list)

# print(len(human_dlist[0]))
# print(human_dlist[0])

# human_dlist[0] = ' '.join(human_dlist[0])
# print(human_dlist[0])                          # so now human_dist[0] became a string

for i in range(len(human_dlist)):
    human_dlist[i] = ' '.join(human_dlist[i])

y_data = human_data.iloc[:, 0].values
# print(y_data)


chimp_dlist = list(chimp_data['Kmers'])
for i in range(len(chimp_dlist)):
    chimp_dlist[i] = ' '.join(chimp_dlist[i])
y_chimp = chimp_data.iloc[:, 0].values                       # y_c for chimp

dog_dlist = list(dog_data['Kmers'])
for i in range(len(dog_dlist)):
    dog_dlist[i] = ' '.join(dog_dlist[i])
y_dog = dog_data.iloc[:, 0].values



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_dlist)
# print(X.shape)
X_chimp = cv.transform(chimp_dlist)
X_dog = cv.transform(dog_dlist)
# print(X_chimp.shape)
# print(X_dog.shape)

# human_data['class'].value_counts().sort_index().plot.bar()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.20, random_state=42)
print(X_train.shape)
print(X_test.shape)

X_chimp_train, X_chimp_test, y_chimp_train, y_chimp_test = train_test_split(X_chimp, y_chimp, test_size = 0.20, random_state=42)
print(X_chimp_train.shape)
print(X_chimp_test.shape)

X_dog_train, X_dog_test, y_dog_train, y_dog_test = train_test_split(X_dog, y_dog, test_size = 0.1, random_state=42)
print(X_dog_train.shape)
print(X_dog_test.shape)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
classifier1 = MultinomialNB(alpha=0.1)
classifier1.fit(X_chimp_train, y_chimp_train)
classifier2 = MultinomialNB(alpha=0.1)
classifier2.fit(X_dog_train, y_dog_train)

y_pred = classifier.predict(X_test)
y_chimp_pred = classifier1.predict(X_chimp_test)
y_dog_pred = classifier2.predict(X_dog_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('\n')
print("Confusion matrix of human data:\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
print('\n')

print("Confusion matrix of chimpanzee data:\n")
print(pd.crosstab(pd.Series(y_chimp_test, name='Actual'), pd.Series(y_chimp_pred, name='Predicted')))
def get_metrics(y_chimp_test, y_chimp_predicted):
    accuracy = accuracy_score(y_chimp_test, y_chimp_predicted)
    precision = precision_score(y_chimp_test, y_chimp_predicted, average='weighted')
    recall = recall_score(y_chimp_test, y_chimp_predicted, average='weighted')
    f1 = f1_score(y_chimp_test, y_chimp_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_chimp_test, y_chimp_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
print('\n')

print("Confusion matrix of dog data:\n")
print(pd.crosstab(pd.Series(y_dog_test, name='Actual'), pd.Series(y_dog_pred, name='Predicted')))
def get_metrics(y_dog_test, y_dog_predicted):
    accuracy = accuracy_score(y_dog_test, y_dog_predicted)
    precision = precision_score(y_dog_test, y_dog_predicted, average='weighted')
    recall = recall_score(y_dog_test, y_dog_predicted, average='weighted')
    f1 = f1_score(y_dog_test, y_dog_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_dog_test, y_dog_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))