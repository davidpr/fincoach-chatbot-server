from fastapi import FastAPI
import llamaindex

app = FastAPI()

# Initialize the Llamaindex library
llamaindex.initialize()

@app.put("/chatbot")
def call_chatbot(input_prompt: str):
    # Call the Llamaindex chatbot with the input prompt
    response = llamaindex.chat(input_prompt)

    return {"response": response}