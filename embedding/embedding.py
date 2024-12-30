from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import uuid
import os, wget

#load the embedding model
embeddings_model = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-30m-english")

#setup the vectordb
db_file = f"embedding/tmp/milvus_{str(uuid.uuid4())[:8]}.db"
print(f"The vector database will be saved to {db_file}")
vector_db = Milvus(embedding_function=embeddings_model, connection_args={"uri": db_file}, auto_id=True)

#load example corpus file
filename = 'embedding/state_of_the_union.txt'
url = 'https://raw.github.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'

if not os.path.isfile(filename):
  wget.download(url, out=filename)

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

#add processed documents to the vectordb
vector_db.add_documents(texts)

#search the vectordb with the query
query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)
print(docs[0].page_content)