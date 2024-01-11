import chromadb
import spacy
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from sentence_transformers import SentenceTransformer

VECTORDB_PATH = './static/vector_storage/'
chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)

# Assuming you have a pre-existing "retrieval_collection"
# collection = chroma_client.get_collection(name="retrieval_collection_cosine")
collection = chroma_client.get_collection(name="retrieval_collection_cosine_mpnet")

# print(collection.peek())
# # print(collection.count())

# embed_model = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-large-en-v1.5-quant",
#         model_kwargs={"device": "cpu"},)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Read CSV file
csv_file_path = './testing.csv'  # Change to the actual path of your CSV file
df = pd.read_csv(csv_file_path)

# Add a new column 'Distance' to store the results
df['Distance'] = ""

# Process each row in the DataFrame
for index, row in df.iterrows():
    try:
        # Extract the input text from the "Modified Question" column
        input_text = row['Modified Question']

        # Skip empty rows
        if pd.isna(input_text) or not input_text.strip():
            print(f"Skipping empty row at index {index}")
            continue

        embedding_list = []

        # embeddings = embed_model.embed_query(input_text)
        # embedding_list.append(embeddings)
        embeddings=model.encode(input_text)
        embeddings_convert = embeddings.tolist()
        embedding_list.append(embeddings_convert)

        # Query ChromaDB collection
        results = collection.query(
            query_embeddings=embedding_list,
            n_results=1
        )

        # Store the distance in the 'Distance' column
        distance_str = str(results['distances']).strip('[[]]')
        df.at[index, 'Distance'] = round(float(distance_str),3)

        # Output the result (modify as needed)
        print(f"Input Text: {input_text}")
        print(results)


        print("---------------------------")

    except Exception as e:
        print(f"Error processing row at index {index}: {repr(e)}")

# Save the updated DataFrame with results to the CSV file
df.to_csv(csv_file_path, index=False)
