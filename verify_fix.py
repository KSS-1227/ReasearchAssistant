import os, re, json
from dotenv import load_dotenv
load_dotenv()

from core.coordinator import ResearchCoordinator
import glob

c = ResearchCoordinator(os.getenv("GEMINI_API_KEY"))
dp = c.document_processor
dp.process_document(glob.glob("data/*.pdf")[0])

raw_docs = dp.documents
step = max(1, len(raw_docs) // 8)
parts = [doc.page_content.strip()[:400] for doc in raw_docs[::step][:8] if doc.page_content.strip()]
paper_text = "\n\n".join(parts)

messages = [
    {"role": "system", "content": "Respond ONLY with a valid JSON array of exactly 6 question strings."},
    {"role": "user",   "content": f"Generate 6 research questions from:\n{paper_text}\nOutput: [\"Q1?\",\"Q2?\",\"Q3?\",\"Q4?\",\"Q5?\",\"Q6?\"]"}
]

resp = c.llm.make_call(messages, json_mode=False)
print("Response:", resp.content[:300] if resp else "None")

if resp:
    raw = re.sub(r'```[a-z]*', '', resp.content).strip().strip('`')
    m = re.search(r'\[[\s\S]*?\]', raw)
    if m:
        qs = json.loads(m.group(0))
        print(f"\nSUCCESS: {len(qs)} questions generated:")
        for i, q in enumerate(qs, 1):
            print(f"  {i}. {q}")
    else:
        print("No JSON array found")
