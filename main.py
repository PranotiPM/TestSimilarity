# This is a sample Python script.

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences)
print(embeddings)

from sentence_transformers import SentenceTransformer, util
class TextSimilar:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embeddings = model.encode(sentences)
    def __init__(self, sentenses, embeddings):
      self.sentenses = sentenses
      self.embeddings = embeddings
    def show(self):
        print(sentences, embeddings)

class query(TextSimilar):
    def __init__(self, query_sentenses, query_embeddings):
      self.query_sentenses = query_sentenses
      self.query_embeddings = query_embeddings
    def show(self):
      print(query_sentenses, query_embeddings)

class passage(TextSimilar):
    def __init__(self, passage_sentenses, passage_embeddings):
      self.passage_sentenses = passage_sentenses
      self.passage_embeddings = passage_embeddings
    def show(self):
      print(passage_sentenses, passage_embeddings)

def cosine_sim(query, passage):
  print("Similarity:", util.cos_sim(query, passage))





