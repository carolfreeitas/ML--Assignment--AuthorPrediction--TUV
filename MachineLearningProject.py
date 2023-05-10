import json
import pandas as pd
import numpy as np
import nltk
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
import string
import re

df = pd.read_json("C:/Users/larsm/OneDrive/Documenten/Tilburg University/MSc Data Science/Machine Learning/Assignment/train.json")
df.head()

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('english')

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import re
import string
from nltk.corpus import stopwords
nltk.download('punkt')

def preprocessing(df_input):
    
    # fill null values in abstract
    df_input['abstract'] = df_input['abstract'].fillna('0')
    
    # merge abstract and title to create new feature
    df_input['all_text'] = df_input['abstract'] + df_input['title']
    df_input['all_text'] = df_input['all_text'].str.lower()

  
    df_input['venue_clean'] = df_input['venue'].apply(clean_venue)
    df_input['venue_clean'] = df_input['venue_clean'].fillna('others')

    df_input['venue_clean'] = df_input['venue'].fillna('others')
    
    df_final = df_input[df_input.columns]
    
    return df_final 

def clean_text(text):
    # remove punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = regex.sub('', text.lower().replace('\n',''))
    
    # remove stopwords and print
    stop_words = stopwords.words('english')
    
    text = ' '.join([w for w in text.split(' ') if not w in stop_words])

    return text



def clean_venue(venue):
    venue = venue.split('@')[-1].lower()
    
    if 'computational linguistics' in venue.lower():
        return 'computational linguistics'
    elif 'workshop on storytelling' in venue.lower():
        return 'workshop on storytelling'
    elif 'nlp4if' in venue.lower():
        return 'nlp4if'
    elif 'vardial' in venue.lower():
        return 'vardial'
    elif 'w-nut' in venue.lower():
        return 'w-nut'
    elif 'sigdial' in venue.lower():
        return 'sigdial'
    elif 'ranlp' in venue.lower():
        return 'ranlp'
    elif 'spanlp' in venue.lower():
        return 'spanlp'
    elif 'anlp' in venue.lower():
        return 'anlp'
    elif 'conll' in venue.lower():
        return 'conll'
    elif 'spnlp' in venue.lower():
        return 'spnlp'
    elif 'nlp4if' in venue.lower():
        return 'nlp4if'
    elif 'emnlp' in venue.lower():
        return 'emnlp'
    elif 'findings' in venue.lower():
        return 'findings'
    elif 'naacl-hlt' in venue.lower():
        return 'naacl-hlt'
    elif 'eacl' in venue.lower():
        return 'eacl'
    elif 'naacl' in venue.lower():
        return 'naacl'
    elif 'acl' in venue.lower():
        return 'acl'
    elif 'cl' in venue.lower():
        return 'cl'
    elif 'semeval' in venue.lower():
        return 'semeval'
    elif 'sem' in venue.lower():
        return 'sem'
    else:
        return 'others'


def lemmatization(tokenized_text):
  text = [wn.lemmatize(c) for c in nltk.word_tokenize(tokenized_text)]
  text = ' '.join(text)
  return text

#import data
df = pd.read_json("C:/Users/larsm/OneDrive/Documenten/Tilburg University/MSc Data Science/Machine Learning/Assignment/train.json")
df_test = pd.read_json("C:/Users/larsm/OneDrive/Documenten/Tilburg University/MSc Data Science/Machine Learning/Assignment/test.json")


#data preprocessing and feature creation
df_processed = preprocessing(df)
df_test_processed = preprocessing(df_test)

#clean text
df['cleaned_context'] = df_processed["all_text"]. apply(lambda x: clean_text(x))
df_test['cleaned_context'] = df_test_processed["all_text"]. apply(lambda x: clean_text(x))

#Train-validation split
X_train, X_val, y_train, y_val = train_test_split(df['cleaned_context'],df_processed['authorId'],test_size=0.2,random_state=42)

#vectorzation
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(X_train)

#Apply transformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

# pipelines with different classifiers
'''classifier = Pipeline([
    ('vect', CountVectorizer(stop_words='english')), 
    ('tfidf', TfidfTransformer()),
    # ('svd', TruncatedSVD(n_components=2000, n_iter=7, random_state=42)),
    ('clf', SGDClassifier(random_state=42, learning_rate='optimal', max_iter=1000, loss='modified_huber')),
])'''

'''classifier = Pipeline([
    ('vect', CountVectorizer(stop_words='english')), 
    ('tfidf', TfidfTransformer()),
    ('randomforest',RandomForestClassifier(n_estimators=100))
])'''

'''classifier = Pipeline([
    ('vect', CountVectorizer(stop_words='english')), 
    ('tfidf', TfidfTransformer()),
    ('nn',MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15,), random_state=1))
])'''

classifier = Pipeline([
    ('vect', CountVectorizer(stop_words='english')), 
    ('tfidf', TfidfTransformer()),
    ('svd', TruncatedSVD(n_components=3000, n_iter=7, random_state=42)),
    ('svc', LinearSVC(C = 7))
])
print('Fitting model')
classifier.fit(X=X_train, y=y_train)
print('Model Fitted')
classifier.score(X_train, y_train)

accuracy_score(y_val, classifier.predict(X_val))

X_test = df_test['cleaned_context'] 

predictions = classifier.predict(X_test)

df_test['authorId'] = predictions

df_test.head()

prediction_list = []
for i in df_test.index:
    predict_dict = {}
    predict_dict['paperId'] = str(df_test['paperId'][i])
    predict_dict['authorId'] = str(df_test['authorId'][i])
    #predict_dict[df_test['paperId'][i]] = str(df_test['authorId'][i])
    prediction_list.append(predict_dict)

with open('predicted.json', 'w') as f:
    json.dump(prediction_list, f)
print('predicted.json created and filled')
