import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

books = pd.read_csv('books_with_categories_and_genre.csv')
books['title'] = books['title'].fillna('')
books['genre'] = books['genre'].fillna('')
books['simple_categories_binary'] = books['simple_categories_binary'].fillna('')
books['description'] = books['description'].fillna('')
books['tagged_description'] = books['isbn13'].astype(str) + ' ' + books['title'] + ' ' + books['genre'] + ' ' + books['simple_categories_binary'] + ' ' + books['description']
books = books[books['tagged_description'].str.strip() != '']
books['tagged_description'].to_csv('tagged_description.txt', sep='\t', index=False, header=False)
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""], strip_whitespace=True)
documents = text_splitter.split_documents(raw_documents)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
class LocalEmbeddings:
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False).tolist()[0]

embedding_model = LocalEmbeddings(embed_model)
db_books = Chroma.from_documents(documents, embedding=embedding_model)
recs = db_books.similarity_search('A suspenseful fiction murder mystery book', k=5)
print('num recs', len(recs))
for r in recs:
    print('---')
    print(r.page_content[:200])
    print('isbn first token:', r.page_content.split()[0])
