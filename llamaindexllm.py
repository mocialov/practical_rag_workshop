from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector
from llama_index.llms.azure_openai import AzureOpenAI

import os
import gc

from dotenv import load_dotenv
load_dotenv()
    
class LLamaIndexLLM():
    def __init__(self):

        llm = AzureOpenAI(
            model="gpt-4o",
            deployment_name=os.getenv('AZURE_CONVERSATIONAL_MODEL_DEPLOYMENT_NAME'),
            api_key=os.getenv("API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("API_VERSION"),
        )

        Settings.llm = llm
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        root_path = "data_encoded"
        data_path = os.path.abspath(root_path)
        subfolders = [f.name for f in os.scandir(data_path) if f.is_dir()]

        self.all_tools = []
        self.all_retrievers = []
        for subfolder in subfolders:

            subsubfolders = [f.name for f in os.scandir(os.path.abspath(os.path.join(root_path, subfolder))) if f.is_dir()]

            for subsubfolder in subsubfolders:
            
                vector_store = FaissVectorStore.from_persist_dir(os.path.join(root_path, subfolder, subsubfolder))
                storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=os.path.join(root_path, subfolder, subsubfolder))
                index = load_index_from_storage(storage_context=storage_context)

                tool = QueryEngineTool.from_defaults(
                    query_engine=index.as_query_engine(similarity_top_k=5), 
                    name=f"company={subfolder}_year=2024_quarter={subsubfolder}",
                    description=f"Company {subfolder} financial report for the 2024 quarter {subsubfolder}"
                )
                self.all_tools.append(tool)

                del vector_store
                del storage_context
                del index
                gc.collect()

    def setup(self):
        self.query_engine = RouterQueryEngine(
            selector=LLMMultiSelector.from_defaults(),
            query_engine_tools=self.all_tools
        )

    def ask(self, question):
        return self.query_engine.query(question).response
    



