import ollama
from ollama import chat
from ollama import ChatResponse
import time

print("Searching models...")

beg1 = time.perf_counter()

models = ollama.list()

end1 = time.perf_counter()

model_names = [model.model for model in models.models]

print(f"Model List fetched in {end1-beg1:.6f} seconds")

options = "\n".join([f"{i+1}. {m}" for i, m in enumerate(model_names)])

choice = int(input(f"\nChoose a model: \n{options} \n Your choice :: "))

folder = r"C:\Users\advai\Documents\Academic\IIRS\Urban LULC Project"

beg2 = time.perf_counter()
response: ChatResponse = chat(model=model_names[choice - 1], messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
end2 = time.perf_counter()

print(response['message']['content'])
print(f"Response fetched in {end2 - beg2:.6f} seconds")