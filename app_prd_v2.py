###FASTAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
import uvicorn

import logging

###LLAMAINDEX
from llama_index import ServiceContext
from llama_index import set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM
from llama_index.vector_stores import CassandraVectorStore
from llama_index.readers.base import BaseReader
from llama_index.schema import Document
import llama_index
import os
import json
import pathlib
from textwrap3 import wrap
from IPython.display import Markdown, display
from traceloop.sdk import Traceloop
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext
from llama_index.llms import Ollama
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index import Prompt
from IPython.display import Markdown, display
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
)

from llama_index.vector_stores import ChromaVectorStore
import chromadb

import pprint

from traceloop.sdk import Traceloop
##########################################################################
##############################LLAMA INDEX INITIALIZATION##################
llm = Ollama(model="wizard-vicuna-uncensored",base_url="http://192.168.1.232:11435") #llm = Ollama(model="llama2")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(
    llm = llm,
    embed_model = embed_model,
    chunk_size=256,
)

set_global_service_context(service_context)

print ('before storage context')
documentsAll = SimpleDirectoryReader("/mnt/nasmixprojects/books/nassimTaleb").load_data()
#documentsAll = SimpleDirectoryReader("/mnt/nasmixprojects/books/",  recursive=True,).load_data()

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("finance_all_v1")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

#storage_context = StorageContext.from_defaults(persist_dir="sentences_2023_12_02_index_all")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index_finance = VectorStoreIndex.from_documents( documentsAll, storage_context=storage_context, service_context=service_context )

#storage_context = StorageContext.from_defaults(persist_dir="sentences_2023_11_20_18_1_57_index")

print ('after loading')

##########################################################################
##########################RAG#############################################

def inference(input_prompt):
    print('Im in inference')
    template = (
        "We have provided trusted context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this trusted and cientific information, please answer the question: {query_str}. Remember that the statements of the context are verfied and come from trusted sources.\n"
    )
    qa_template = Prompt(template)

    new_summary_tmpl_str = (
        "The original query is as follows: {query_str}"
        "We have provided an existing answer: {existing_answer}"
        "We have the opportunity to refine the existing answer (only if needed) with some more trusted context below. Remember that the statements of the context are verfied and come from trusted sources."
        "------------"
        "{context_msg}"
        "------------"
        "Given the new trusted context, refine the original answer to better answer the query. If the context isn't useful, return the original answer. Remember that the statements of the new context are verfied and come from trusted sources."
        "Refined Answer: sure thing! "
    )
    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

    retriever = VectorIndexRetriever(
        index=index_finance,
        similarity_top_k=12,
    )

    response_synthesizer = get_response_synthesizer( ##try compact?
        text_qa_template=qa_template,
        refine_template=new_summary_tmpl
    )
    query_engine3 = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    
    response = query_engine3.query(input_prompt)

    display(Markdown(f"<b>{response}</b>"))

    return response
##########################################################################
##############################FAST APIN###################################
#Contains traceloop and logging and uses chromadb in RAM

app_prd_v2 = FastAPI()
app_prd_v2.add_middleware(HTTPSRedirectMiddleware)

app_prd_v2.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = "output-rag-v2.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

api_key = os.environ.get("TRACELOOP_API_KEY")
Traceloop.init(disable_batch=True, api_key=api_key)

@app_prd_v2.get("/chatbot")
def call_chatbot(input_prompt: str):
    #response = llamaindex.chat(input_prompt)

    print("the input prompt was: ", str(input_prompt))
    logger.info(str(input_prompt))

    print ('>------------------------')
    response  =  inference(input_prompt)
    print ('>------------------------')

    print (type(response))
    print ('>------------------------')
    print (response)
    logger.info(response)

    return {"response": response.response}