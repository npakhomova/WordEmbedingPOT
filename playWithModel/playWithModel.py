from gensim.models import Word2Vec
from sklearn.cluster import KMeans


def nicePrint(dictionary):
    print ('----------------')
    for row in dictionary:
        print " %s " % row[0] + ': {0:f}'.format(row[1])


model = Word2Vec.load("../data/dresses_stemmed_bigrams_long_description_model")

nicePrint( model.most_similar(positive=["winter", "warm"], negative=["summer"], topn=10))

nicePrint( model.most_similar(positive=["prom"], topn=10))


