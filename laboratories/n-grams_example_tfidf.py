from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
corpus = ["buena pelicula", "no me gusto la pelicula", "no me gusto", "me gusto la pelicula", "buena"]

# using default tokenizer in TfidfVectorizer
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
features = tfidf.fit_transform(corpus)
df = pd.DataFrame(features.toarray(),
                  index=['sentence '+str(i) for i in range(1, 1+len(corpus))],
                  columns=tfidf.get_feature_names())
print(df.to_string())