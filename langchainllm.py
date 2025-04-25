from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain.schema import AgentFinish
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import AgentExecutor
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate

import os
import gc

from dotenv import load_dotenv
load_dotenv()

class FinancialAnalystPrompt:
    def __init__(self, all_tools):
        self.prompt = PromptTemplate.from_template("""
        You are a great financial analyst that is amazing at analysing annual reports. 
        Use revenue, profit, EBITDA, income, cash flow, earnings to answer any questions whether technical or general.
        Answer the following questions performing proper research and using all tools at your disposal. 
        You have access to the following tools: {tools}

        Use each tool at least once to ensure a comprehensive analysis.

        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin! Reminder that you should use all the tools available to you.

        Question: {input}
        Thought:{agent_scratchpad}
        """)

        self.prompt = self.prompt.partial(
            tools=render_text_description(list(all_tools)),
            tool_names=", ".join([t.name for t in all_tools]),
        )

    def get_prompt(self):
        return self.prompt

def simple_parser(output):
    return AgentFinish(return_values={"output": output.content}, log=output.content)

class LangChainLLM():
    def __init__(self):

        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        root_path = "data_encoded"
        data_path = os.path.abspath(root_path)
        subfolders = [f.name for f in os.scandir(data_path) if f.is_dir()]

        self.all_tools = []
        for subfolder in subfolders:

            subsubfolders = [f.name for f in os.scandir(os.path.abspath(os.path.join(root_path, subfolder))) if f.is_dir()]

            for subsubfolder in subsubfolders:
            
                vector_store = FaissVectorStore.from_persist_dir(os.path.join(root_path, subfolder, subsubfolder))
                storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=os.path.join(root_path, subfolder, subsubfolder))
                index = load_index_from_storage(storage_context=storage_context)

                query_engine = index.as_query_engine()

                tool_config = IndexToolConfig(
                    query_engine=query_engine,
                    name=f"Company {subfolder} financial report for the 2024 quarter {subsubfolder}",
                    description=f"helps answer queries about {subfolder} quarter {subsubfolder} report for year 2024"
                )

                tool = LlamaIndexTool.from_tool_config(tool_config)

                self.all_tools.append(tool)

                del vector_store
                del storage_context
                del index
                del query_engine
                del tool_config
                gc.collect()

    def setup(self):

        llm = AzureChatOpenAI(
            azure_deployment=os.getenv('AZURE_CONVERSATIONAL_MODEL_DEPLOYMENT_NAME'),  # or your deployment
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("API_KEY"),
            temperature=0.0
        )

        self.agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | FinancialAnalystPrompt(self.all_tools).get_prompt()
            | llm
            | RunnableLambda(simple_parser)
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.all_tools, 
            return_intermediate_steps=True, 
            verbose=False, 
            handle_parsing_errors=True, 
            max_iterations=10
        )

        self.final_chain = self.agent_executor

    def ask(self, question):
        result = self.final_chain.invoke(
            {
                "input": question,
                "chat_history": []
            })

        return result["output"]