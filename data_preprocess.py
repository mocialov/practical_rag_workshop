# from thinc.api import set_gpu_allocator

# set_gpu_allocator("pytorch")

import os
import json

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (VectorStoreIndex, StorageContext, Document, Settings)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import faiss

import spacy
#spacy.require_cpu()
from spacy_layout import spaCyLayout

import pandas as pd

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def dataframe_similarity(df1, df2):
    if df1.shape != df2.shape:
        return 0
    matches = (df1.values == df2.values).sum()
    total_elements = df1.size
    return matches / total_elements

def get_text_data(doc_):
    text_data = {}
    for span in doc_.spans["layout"]:
        if span.label_ in ["text", "list_item", "paragraph"]:
            text_data[str(span._.heading)] = span.text
    return text_data

def get_table_data(doc_):
    unique_dfs_heading = []
    df_row_heading = {}

    for table in doc_._.tables:
        heading = table._.heading.text
        df = table._.data
        is_unique = True
        for unique_df_heading in unique_dfs_heading:
            similarity = dataframe_similarity(df, unique_df_heading[1])
            if similarity >= 0.6:
                is_unique = False
                break
        if is_unique:

            unique_dfs_heading.append((heading, df))

            df.columns = pd.io.common.dedup_names(df.columns, is_potential_multiindex=False)

            df = df.replace("", float("NaN"))
            df = df.dropna(thresh=len(df.columns)/2)

            rows = df.to_dict(orient='records')

            df_row_heading[heading] = rows

    return df_row_heading

def make_llama_documents(text_data, df_row_heading, subfolder, file_name):
    documents = []
    for key in df_row_heading.keys():

        # if len(df_row_heading[key]) > 0:
        #     with open(f"tables.txt", "a+") as f:
        #         f.write(json.dumps({
        #             "table_name": key,
        #             "columns": list(df_row_heading[key][0].keys()),
        #             "rows": [list(item_.values()) for item_ in df_row_heading[key]]
        #         }, indent=4)+"\n")

        if len(df_row_heading[key]) > 0:
            documents.append(
                Document(
                    text=json.dumps({
                        "table_name": key,
                        "columns": list(df_row_heading[key][0].keys()),
                        "rows": [list(item_.values()) for item_ in df_row_heading[key]]
                    }), 
                    metadata={
                        "company": subfolder,
                        "doc_id": "table_"+key,
                        "source": file_name.split(".")[0],
                        "year": file_name.split("_")[-1].split(".")[0]
                        }
                    )
            )

    for key in text_data.keys():
        # with open(f"texts.txt", "a+") as f:
        #     f.write(json.dumps({key:text_data[key]})+"\n")

        documents.append(
            Document(
                text=json.dumps({key:text_data[key]}), 
                metadata={
                    "company": subfolder,
                    "doc_id": "text_"+key,
                    "source": file_name.split(".")[0],
                    "year": file_name.split("_")[-1].split(".")[0]
                    }
                )
        )
    
    return documents


data_path = "data"
subfolders = [f.name for f in os.scandir(data_path) if f.is_dir()]

all_documents = []
for subfolder in subfolders:
    print (subfolder)

    file_names = [f for f in os.listdir(os.path.join(data_path, subfolder)) if os.path.isfile(os.path.join(data_path, subfolder, f))]

    for file_name in file_names:
        print (file_name)

        layout = spaCyLayout(spacy.blank("en"))
        doc = layout(os.path.join(data_path, subfolder, file_name))

        text_data = get_text_data(doc)
        df_row_heading = get_table_data(doc)

        documents = make_llama_documents(text_data, df_row_heading, subfolder, file_name)

        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(384))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist(os.path.join("data_encoded", subfolder, file_name.split("_")[-1].split(".")[0]))