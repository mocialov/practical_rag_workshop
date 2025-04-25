from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.base_retriever import BaseRetriever

import dspy

import os
import gc
import json
from typing import List

from dotenv import load_dotenv
load_dotenv()

class LlamaIndexRetriever(dspy.Retrieve):

    retriever: BaseRetriever

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def forward(self, query):
        raw = self.retriever.retrieve(query)

        reponse = []
        for item in raw:
            try:
                reponse.append(list(json.loads(item.text).values())[0])
            except:
                pass

        return reponse

class BestDataSourceGuesser(dspy.Signature):
    """
    You are an expert information retrieval specialist tasked with evaluating and selecting the most relevant information from multiple data sources. 
    Your primary responsibility is to analyze the user's question and match it with the most appropriate retrieved information 

    When evaluating the responses, consider:
    - Relevance to the user's question
    - Accuracy and completeness of the information
    - Reliability of the source
    """
    question: str = dspy.InputField(desc="The specific query or question from the user requiring information")
    retriever_information: List[str] = dspy.InputField(desc="Descriptions of available data sources and their characteristics")
    answers: List[str] = dspy.InputField(desc="Responses obtained from different data sources")
    best_answer_number: int = dspy.OutputField(desc="A single numerical response indicating the best answer number from the retrieved answers that most appropriately addresses the user's question.")

class DataSourceRetrieverSummarizer(dspy.Signature):
    """You are very good at summarizing multiple sentences into a paragraph given user question and a list of sentences.
    Make sure to summarize the sentences into a paragraph that answers the user question.
    """
    question: str = dspy.InputField(desc="user question")
    answers: str = dspy.InputField(desc="a list of sentences")
    summary: str = dspy.OutputField(desc="summary of the answers")

class DSPyLLM():
    def __init__(self):

        self.llm = dspy.LM(api_base=os.getenv("AZURE_ENDPOINT"),
                api_version=os.getenv("API_VERSION"),
                api_key=os.getenv("API_KEY"),
                model=f"azure/{os.getenv('AZURE_CONVERSATIONAL_MODEL_DEPLOYMENT_NAME_JUDGE')}",
                temperature=0.0,
                cache=False)
        
        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        root_path = "data_encoded"
        data_path = os.path.abspath(root_path)
        subfolders = [f.name for f in os.scandir(data_path) if f.is_dir()]

        self.all_query_engines = []

        for subfolder in subfolders:

            subsubfolders = [f.name for f in os.scandir(os.path.abspath(os.path.join(root_path, subfolder))) if f.is_dir()]

            for subsubfolder in subsubfolders:
            
                vector_store = FaissVectorStore.from_persist_dir(os.path.join(root_path, subfolder, subsubfolder))
                storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=os.path.join(root_path, subfolder, subsubfolder))
                index = load_index_from_storage(storage_context=storage_context)

                self.all_query_engines.append(
                    (f"Company {subfolder} financial report for the 2024 quarter {subsubfolder}", 
                     LlamaIndexRetriever(index.as_retriever(similarity_top_k=5)))
                )

                del vector_store
                del storage_context
                del index
                gc.collect()

    def setup(self):
        self.generate_answer = dspy.Predict(BestDataSourceGuesser)
        self.summarize_answer = dspy.ChainOfThought(DataSourceRetrieverSummarizer)

    def ask(self, question):

        with dspy.context(lm=self.llm):
            retriever_answers = [retriever(question) for (retriever_info, retriever) in self.all_query_engines]

            best_retriever = self.generate_answer(question=question,
                                                  retriever_information=[retriever_info for (retriever_info, retriever) in self.all_query_engines],
                                                  answers=retriever_answers).best_answer_number - 1
            
            return self.summarize_answer(question=question,
                                         answers=retriever_answers[best_retriever]).summary
    



