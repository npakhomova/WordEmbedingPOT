from textpreprocessing import processSentenceFromFiles
from gensim.models import word2vec
from gensim.models import Phrases

# important: parameters to play with!!!

# hyperparameters for w2vec model
num_features = 300    # Word vector dimensionality
min_word_count = 100   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# hyperparameters for Phrases model

# see gensim.model.Phrases doc: `threshold` represents a threshold for forming the phrases (higher means
# fewer phrases). A phrase of words `a` and `b` is accepted if
# `(cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold`, where `N` is the
# total vocabulary size.
THRESHOLD = 5.0
threeTermsMinFreq = 10
biTermsMinFreq = 30

print "Preprocessing text..."
sentences = processSentenceFromFiles("../data/dresses.txt")

print "Mining bigrams..."
#if some 2 terms collocation meets more then 40 times then it processed as bgramm
biTerms = Phrases(sentences=sentences, min_count=biTermsMinFreq, threshold=THRESHOLD)

print "Mining threegrams..."
#if some 3 terms collocation meets more then 10 times then it processed as bgramm
threeTerms = Phrases(sentences=biTerms[sentences], min_count=threeTermsMinFreq, threshold=THRESHOLD)

print "Train model..."
#train model
model = word2vec.Word2Vec(threeTerms[sentences], workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)

#save model
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "../data/dresses_stemmed_bigrams_long_description_model"
model.save(model_name)