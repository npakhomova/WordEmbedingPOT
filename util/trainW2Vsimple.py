from textpreprocessing import processSentenceFromFiles
from gensim.models import word2vec

# IMPORTANT! THIS is parameters to play with!!!
# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words


print "Preprocessing text..."
sentences = processSentenceFromFiles("../data/dresses.txt")


print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# save the model for later use. You can load it later using Word2Vec.load()
model_name = "../data/dresses_stemmed_long_description_model"
model.save(model_name)