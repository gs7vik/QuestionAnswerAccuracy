from langchain.embeddings import HuggingFaceEmbeddings
from scipy.spatial.distance import cosine
embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"})

question1="What is the stock price of Shree Cement in India?"
question2="what is the stock price of Shree Cement "
embedding_que1=embed_model.embed_query(question1)
embedding_que2=embed_model.embed_query(question2)
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine(embedding1, embedding2)
ans=calculate_cosine_similarity(embedding_que1,embedding_que2)
print(round(float(ans),3))