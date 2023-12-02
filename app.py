from fastapi import FastAPI
#import llamaindex

app = FastAPI()

# Initialize the Llamaindex library
#llamaindex.initialize()

@app.get("/chatbot")
def call_chatbot(input_prompt: str):
    # Call the Llamaindex chatbot with the input prompt
    #response = llamaindex.chat(input_prompt)
    print("the input prompt was: ", str(input_prompt))
    response = "The response from the Llamaindex chatbot is: hi"
    return {"response": response}