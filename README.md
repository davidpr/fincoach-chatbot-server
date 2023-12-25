# Fincoach Chatbot Server
This is a fastAPI server calling a LLamaIndex chatbot

## Getting started
FastAPI env.

```
python3 -m venv v-fincoach-fastapi-llamaindex
source v-fincoach-fastapi-llamaindex/bin/activate
pip install fastapi uvicorn[standard]
(optional python-multipart, sqlalquemy jina2)

pip install llama_index
pip install textwrap3
pip install IPython
pip install traceloop-sdk
pip install langchain
pip install transformers
pip install torch

pip install EbookLib html2text
pip install pypdf
pip install chromadb

```

Trulens env
```
cd /home/david/weaviate-tests/weaviate-videorack
source v-translations-llama/bin/activate

pip install trulens-eval


traceloop-sdk 0.3.5 requires pydantic<3.0.0,>=2.5.0, but you have pydantic 1.10.13 which is incompatible.


pip install litellm


```


## Using the server

- [ ] uvicorn app:app --reload
- [ ] uvicorn app:app --host 0.0.0.0 --port 9999 --reload
- [ ] uvicorn app:app --host 0.0.0.0 --port 9999 --workers 16
- [ ] uvicorn app:app --host 0.0.0.0 --port 9999 --ssl-keyfile=/home/david/weaviate-tests/fincoach-chatbot-server/privkey3.pem --ssl-certfile=/home/david/weaviate-tests/fincoach-chatbot-server/fullchain3.pem
- [ ] Production: 
- [ ] uvicorn app_prd:app_prd --host 0.0.0.0 --port 9999 --ssl-keyfile=/home/david/weaviate-tests/fincoach-chatbot-server/privkey3.pem --ssl-certfile=/home/david/weaviate-tests/fincoach-chatbot-server/fullchain3.pem
- [ ] nohup uvicorn app_prd:app_prd --host 0.0.0.0 --port 9999 --ssl-keyfile=/home/david/weaviate-tests/fincoach-chatbot-server/privkey3.pem --ssl-certfile=/home/david/weaviate-tests/fincoach-chatbot-server/fullchain3.pem > output_all_rag.txt &


nohup uvicorn app_prd_v2:app_prd_v2 --host 0.0.0.0 --port 9998 --ssl-keyfile=/home/david/weaviate-tests/fincoach-chatbot-server/privkey3.pem --ssl-certfile=/home/david/weaviate-tests/fincoach-chatbot-server/fullchain3.pem > output-all-rag-v2.txt 2>&1 & 

## More info about uvicorn
- [ ] https://fastapi.tiangolo.com/deployment/manually/
uvicorn vs gunicorn WSGI ASGI & parallel workers
https://fastapi.tiangolo.com/deployment/concepts/
https://fastapi.tiangolo.com/deployment/server-workers/


***


## Make testing in the jupyter

jupyter2023.

