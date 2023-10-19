import spacy
from gensim.models import Word2Vec

# Define a function to process the text and extract tokens
def process_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return tokens

# Define your dataset as a list of sentences
sentences = [
    "What is Terrence Ross' nationality",
    "What clu was in Toronto 1995-96",
    # ... Add all your questions here ...
]

# Tokenize the sentences
tokenized_sentences = [process_text(sentence) for sentence in sentences]

# Train a Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

# Save the model
# model.save("word2vec_model")

# Now, you can use the model to get word embeddings for each word in your dataset.
# For example, to get the word embedding for "Terrence", you can use:
vector = model.wv['Terrence']
print(vector)
