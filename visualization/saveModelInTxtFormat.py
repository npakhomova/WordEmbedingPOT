from gensim.models import Word2Vec




model = Word2Vec.load("../data/dresses_stemmed_bigrams_long_description_model")
model.save_word2vec_format('../data/dresses_long_description_vector.txt', binary=False)

outfile = open("../data/dresses_long_description_dictionary.txt","w")
outfile.write("\n".join(list(model.vocab)))