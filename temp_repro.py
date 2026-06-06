import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print('imports ok')
books = pd.read_csv('books_with_categories_and_genre.csv')
print('csv loaded', books.shape)
print('isbn13 dtype', books['isbn13'].dtype)
print(books['isbn13'].head())
books['title'] = books['title'].fillna('')
books['genre'] = books['genre'].fillna('')
books['simple_categories_binary'] = books['simple_categories_binary'].fillna('')
books['description'] = books['description'].fillna('')
books['tagged_description'] = books['isbn13'].astype(str) + ' ' + books['title'] + ' ' + books['genre'] + ' ' + books['simple_categories_binary'] + ' ' + books['description']
books = books[books['tagged_description'].str.strip() != '']
books['tagged_description'].to_csv('tagged_description.txt', sep='\t', index=False, header=False)
print('tagged_description created', len(books))
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
print('raw_documents loaded', len(raw_documents))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""], strip_whitespace=True)
documents = text_splitter.split_documents(raw_documents)
print('documents split', len(documents))
