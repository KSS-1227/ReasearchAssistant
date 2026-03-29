with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = "        resp = coordinator.llm.make_call(messages)\n"
new = "        resp = coordinator.llm.make_call(messages, json_mode=False)\n"

if old in content:
    content = content.replace(old, new, 1)
    print("json_mode fix: OK")
else:
    print("NOT FOUND")

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done.")
