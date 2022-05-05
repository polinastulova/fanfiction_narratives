f = open("phantom_temp.txt", "r", encoding = "utf-8-sig")
phantom_original = f.read()

phantom_lower = phantom_original.lower()

from gensim.parsing.preprocessing import remove_stopwords
phantom_without_stopwords = remove_stopwords(phantom_lower)
print(phantom_without_stopwords)

docs = [phantom_without_stopwords[start:start+3000] for start in range(0, len(phantom_without_stopwords), 3000)]
#docs = phantom_without_stopwords.split("chapter")

# Tokenize the documents.
from nltk.tokenize import RegexpTokenizer

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    print(docs[idx][:10])
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    print(docs[idx][:10])
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
    print(docs[idx][:10])

docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

docs = [[token for token in doc if len(token) > 1] for doc in docs]

from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

from gensim.models import Phrases

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in docs]

from gensim.models import LdaModel

# Set training parameters.
num_topics = 20
chunksize = 1500
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)
