import os, time, dotenv
os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = 'YES'
dotenv.load_dotenv()
from google import genai
from google.genai import types
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
context = ['On 01.07.2021, the petitioners sold equity shares of company under Offer For Sale to the public. Pramod Swarup Agrawal transferred 11,11,486 equity shares for Rs.33,00,00,000/-. Sneh Lata Agarwal transferred 14,05,405 equity shares for Rs.41,59,99,880/-.']
prompt = FaithfulnessTemplate.generate_truths(retrieval_context='\n\n'.join(context), extraction_limit=10)

print('Sending generate_truths prompt to gemma-4-31b-it...', flush=True)
t0 = time.time()
resp = client.models.generate_content(
    model='gemma-4-31b-it',
    contents=prompt + '\n\nRespond strictly in valid JSON adhering to schema: {"truths": ["..."]}',
    config=types.GenerateContentConfig(temperature=0.0)
)
print(f'Done in {time.time() - t0:.2f}s:\n{resp.text}', flush=True)
