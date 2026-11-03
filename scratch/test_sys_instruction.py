import os, time, dotenv
from google import genai
from google.genai import types

dotenv.load_dotenv()
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

for sys_inst in [None, 'You are an evaluator.']:
    t0 = time.time()
    try:
        cfg = types.GenerateContentConfig(temperature=0.0) if not sys_inst else types.GenerateContentConfig(temperature=0.0, system_instruction=sys_inst)
        resp = client.models.generate_content(
            model='gemma-4-31b-it',
            contents='Return JSON: {"status": "ok"}',
            config=cfg
        )
        print(f"sys_inst={sys_inst} -> SUCCESS in {time.time()-t0:.2f}s: {resp.text.strip()}", flush=True)
    except Exception as e:
        print(f"sys_inst={sys_inst} -> ERROR in {time.time()-t0:.2f}s: {e}", flush=True)
