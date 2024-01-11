import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from scipy.spatial.distance import cosine

# Assuming you have loaded your CSV file into a pandas DataFrame
df = pd.read_csv('./small_Train_100.csv',encoding='cp1252')

# Define the Hugging Face Embeddings model
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"}
)

# Function to calculate cosine similarity
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine(embedding1, embedding2)

# Function to apply the embedding and calculate cosine distance
def process_row(row):
    embedding_que1 = embed_model.embed_query(row['question1'])
    embedding_que2 = embed_model.embed_query(row['question2'])
    cos_distance = calculate_cosine_similarity(embedding_que1, embedding_que2)
    return round(float(cos_distance), 3)

# Apply the function to calculate cosine distance and create 'cosine_distance' column
df['cosine_distance'] = df.apply(process_row, axis=1)

# Assign 'dup' column based on cosine distance threshold
threshold = 0.1
df['dup'] = df['cosine_distance'].apply(lambda x: 1 if x <= threshold else 0)

# Save the updated DataFrame to the same CSV file
df.to_csv('your_file.csv', index=False)
