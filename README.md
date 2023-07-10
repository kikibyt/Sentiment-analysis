# Sentiment-analysis
Certainly! Here's an example of a README file for the project:

# Sentiment Analysis Project - University of Exeter Reviews

This project implements sentiment analysis on a dataset of reviews from the University of Exeter. The goal is to classify the sentiment of the reviews as positive or negative.

## Project Overview

The project performs the following tasks:

1. **Data Preprocessing:** The dataset is read from a CSV file and duplicates are removed. The reviews are cleaned by removing HTML tags, apostrophes, alphanumeric words, and special characters. Text is lowercased, tokenized, and subjected to stopword removal and lemmatization.

2. **Document Corpus Creation:** A document corpus is created by applying the text cleaning process to each review. The corpus serves as the input for further analysis.

3. **Feature Extraction:** The Bag of Words model (CountVectorizer) is used to convert the text data into numerical features. The corpus is transformed into a matrix of word counts.

4. **Model Training and Evaluation:** The dataset is split into training and test sets. A Naive Bayes classifier is trained on the training data. The trained model is then used to predict the sentiment of new reviews.

5. **User Interaction:** The user can enter new reviews, and the trained model predicts their sentiment as positive or negative.

## Requirements

The following libraries and resources are required to run the project:

- pandas
- BeautifulSoup
- re
- nltk
- scikit-learn

You also need to download NLTK resources, specifically the stopwords and wordnet.

## Usage

1. Install the required libraries using pip or any package manager.

2. Download the NLTK resources by running the following commands:
   ```
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. Place the dataset file (`ExeterReviews.csv`) in the project directory.

4. Run the code file (`sentiment_analysis.py`) using a Python interpreter.

5. Follow the prompts to interact with the sentiment analysis system and enter new reviews.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to use and modify the code according to your needs.

## Acknowledgements

The dataset used in this project was obtained from [https://www.kaggle.com/datasets/rohitpawar1/university-of-exeter-reviews].

If you have any questions or suggestions, feel free to reach out to [mercyokebiorun@gmail.com].

Happy sentiment analysis!
