from core.llm_interface import LLMInterface

api_key = 'AIzaSyBqkIwHSMdQBP80u-W2Qh_R51zOEe7_hT0'
llm = LLMInterface(api_key=api_key)

messages = [
    {
        'role': 'system',
        'content': 'Respond ONLY with a valid JSON array of exactly 6 question strings. No markdown, no explanation.'
    },
    {
        'role': 'user',
        'content': 'Generate exactly 6 specific research questions from this paper content. Return ONLY a JSON array: ["Q1?","Q2?","Q3?","Q4?","Q5?","Q6?"]\n\nPaper:\nThis is a test abstract about deep learning and transformers.'
    }
]

resp = llm.make_call(messages, json_mode=False)
print('LLM response:', getattr(resp, 'content', None))
