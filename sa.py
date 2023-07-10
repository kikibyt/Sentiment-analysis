# Importing the libraries
from builtins import input
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Importing the dataset
dataset = pd.read_csv('ExeterReviews.csv')


# Dropping duplicates in the dataset
dataset = dataset.drop_duplicates(subset=["Member ID", "Date", "Review"], keep='first', inplace=False)

# Function to remove HTML tags from review
def removeHTMLTags(review):
    soup = BeautifulSoup(review, 'lxml')
    return soup.get_text()

# Function to remove apostrophes from words
def removeApostrophe(review):
    phrase = re.sub(r"won't", "will not", review)
    phrase = re.sub(r"can't", "can not", review)
    phrase = re.sub(r"n't", " not", review)
    phrase = re.sub(r"'re", " are", review)
    phrase = re.sub(r"'s", " is", review)
    phrase = re.sub(r"'d", " would", review)
    phrase = re.sub(r"'ll", " will", review)
    phrase = re.sub(r"'t", " not", review)
    phrase = re.sub(r"'ve", " have", review)
    phrase = re.sub(r"'uni", " University", review)
    return phrase

# Function to remove alphanumeric words from review
def removeAlphaNumericWords(review):
    return re.sub("\S*\d\S*", "", review).strip()

# Function to remove special characters from review
def removeSpecialChars(review):
    return re.sub('[^a-zA-Z]', ' ', review)

# Function to partition scores into positive (1) and negative (0)
def scorePartition(x):
    if x < 3:
        return 0
    return 1

# Function to perform text cleaning


    # Rest of the cleaning steps
    ...

def doTextCleaning(review):
    if pd.isnull(review) or isinstance(review, float):
        return ""
    review = removeHTMLTags(review)
    review = removeApostrophe(review)
    review = removeAlphaNumericWords(review)
    review = removeSpecialChars(review) 
    # Lower casing
    review = review.lower()  
    # Tokenization
    review = review.split()
    # Removing Stopwords and Lemmatization
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')  # Example: retain 'not' as it can be important for sentiment analysis

    lmtzr = WordNetLemmatizer()
    review = [lmtzr.lemmatize(word, 'v') for word in review if not word in stop_words]

    review = " ".join(review)    
    return review

# Generalizing the score
actualScore = dataset['Student Review']
positiveNegative = actualScore.map(scorePartition) 
dataset['Score'] = positiveNegative

# Creating the document corpus
corpus = []   
for index, row in tqdm(dataset.iterrows()):
    review = doTextCleaning(row['Review'])
    corpus.append(review)

for doc in corpus:
    if len(doc) == 0:
        print("Empty Document")
    else:
        print("Non-Empty Document")


print("Number of documents:", len(corpus))
print("Sample documents:")
for doc in corpus[:5]:
    print("-", doc)

    # Check if all documents are empty
empty_documents = [doc for doc in corpus if not doc]
print("Number of empty documents:", len(empty_documents))



# Creating the Bag of Words model
cv = CountVectorizer(ngram_range=(1,3), max_features=5000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Function to predict the sentiment for a new review
def predictNewReview():
    input("Press Enter to exit...")

    newReview = input("Type the Review: ")
    
    if newReview == '':
        print('Invalid Review')  
    else:
        cleanedReview = doTextCleaning(newReview)
        if cleanedReview == '':
            print('Invalid Review after cleaning')  
        else:
            new_review = cv.transform([cleanedReview]).toarray()  
            prediction = classifier.predict(new_review)
            if prediction[0] == 1:
                print("Positive Review")
            else:        
                print("Negative Review")

    
    
# Test the model on new reviews
predictNewReview()
