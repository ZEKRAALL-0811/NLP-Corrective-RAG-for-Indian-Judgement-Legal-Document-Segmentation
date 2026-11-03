import os, time, dotenv
from google import genai
from google.genai import types
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate

dotenv.load_dotenv()
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

truths = ['On 01.07.2021, the petitioners sold equity shares of company under Offer For Sale to the public.', 'Pramod Swarup Agrawal transferred 11,11,486 equity shares for Rs.33,00,00,000/-.']
claims = ['Pramod Swarup Agrawal transferred 11,11,486 equity shares for a total consideration of Rs.33,00,00,000/- on 01.07.2021.']

prompt = FaithfulnessTemplate.generate_verdicts(claims=claims, retrieval_context='\n\n'.join(truths))
prompt = prompt.replace('otherwise I WILL DIE.', '')
prompt += '\n\nIMPORTANT: Respond strictly in valid JSON format only.'

safety_settings = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
]

config = types.GenerateContentConfig(
    temperature=0.0,
    safety_settings=safety_settings,
)

t0 = time.time()
print('Calling gemma-4-31b-it with safety_settings BLOCK_NONE...', flush=True)
resp = client.models.generate_content(
    model='gemma-4-31b-it',
    contents=prompt,
    config=config
)
print(f'Response in {time.time() - t0:.2f}s:\n{resp.text}', flush=True)
