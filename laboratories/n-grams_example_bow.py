from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
corpus = ["buena pelicula", "no me gusto la pelicula", "no me gusto", "me gusto la pelicula", "buena"]

bow = CountVectorizer(analyzer='word', ngram_range=(1, 1))
features = bow.fit_transform(corpus)

df = pd.DataFrame(features.toarray(),
                  index=['document '+str(i) for i in range(1, 1+len(corpus))],
                  columns=bow.get_feature_names())
print(df.to_string())

