
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from unidecode import unidecode
from gensim.parsing import PorterStemmer

#actually don't like this stemmer. neccessary to find smth other
stemmer = PorterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



def remove_non_ascii(text):
    return unidecode(unicode(text, encoding = "utf-8"))


def stem(word):
    return stemmer.stem(word)


def sentence_to_wordlist(text, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 2. Remove non-letters
    result_text = re.sub("[^a-zA-Z]"," ", text)
    #
    # 3. Convert words to lower case and split them
    words = result_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 4.1 stem words
    if (stem):
        words = [stem(word) for word in words]
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences(paragraph, tokenizer, remove_stopwords=True, stem=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 0. Remove non asci symbols
    raw_sentences = remove_non_ascii(paragraph)
    # remove html like symbols
    raw_sentences = BeautifulSoup(raw_sentences).get_text()
    raw_sentences = re.sub(r'[^\x00-\x7F]+', ' ', raw_sentences)
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(raw_sentences.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(sentence_to_wordlist(raw_sentence, remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences




# read data from file. Expect that 1 lina couls contains several setnceces
# return preprocessed sentences (stop words/stemming)
def processSentenceFromFiles(fname, stem=False):
    sentences = []  # Initialize an empty list of sentences

    with open(fname) as f:
        unlabeled_dresses = f.readlines()
    for dressDescription in unlabeled_dresses:
        if (dressDescription.__len__() >= 0):
            sentences += review_to_sentences(dressDescription, tokenizer, stem=stem)
    return sentences

