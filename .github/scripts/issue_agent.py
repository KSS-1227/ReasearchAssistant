import os
import re
import json
import base64
import requests

# ── Config from environment ──────────────────────────────────────────────
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPO  = os.environ["GITHUB_REPO"]
ISSUE_NUMBER = int(os.environ["ISSUE_NUMBER"])
LLM_API_KEY  = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL    = os.environ.get("LLM_MODEL", "meta/llama-3.3-70b-instruct")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

# ── Step 1: Read the issue ───────────────────────────────────────────────
def get_issue():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{ISSUE_NUMBER}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    print(f"✅ Issue #{ISSUE_NUMBER}: {data['title']}")
    return data["title"], data.get("body") or ""

# ── Step 2: Read the codebase ────────────────────────────────────────────
def get_file(path, branch="main"):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    r = requests.get(url, headers=HEADERS, params={"ref": branch}, timeout=30)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return content, data["sha"]

def get_repo_tree():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/main?recursive=1"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    files = [
        f["path"] for f in r.json().get("tree", [])
        if f["type"] == "blob" and f["path"].endswith(".py")
        and not f["path"].startswith(".github")
    ]
    print(f"✅ Found {len(files)} Python files")
    return files[:8]  # limit to avoid token overflow

# ── Step 3: Ask LLM to fix the issue ────────────────────────────────────
def ask_llm(issue_title, issue_body, files_content):
    prompt = f"""You are an expert Python developer. Fix the following GitHub issue.

ISSUE TITLE: {issue_title}

ISSUE DESCRIPTION:
{issue_body}

CODEBASE:
{files_content}

YOUR TASK:
- Identify which file needs to be changed
- Write the COMPLETE fixed version of ONLY that one file
- Respond with ONLY a valid JSON object — nothing else

JSON FORMAT (copy this exactly):
{{"file_path": "filename.py", "fixed_content": "full file content here with \\n for newlines", "explanation": "brief description of change"}}

STRICT RULES:
- Output ONLY the JSON object
- No markdown, no code fences, no backticks
- No literal newlines inside JSON string values — use \\n
- Escape all double quotes inside strings as \\"
- Start response with {{ and end with }}"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.1
    }

    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=120
    )
    r.raise_for_status()

    raw = r.json()["choices"][0]["message"]["content"]
    print(f"✅ LLM responded ({len(raw)} chars)")
    print(f"📄 Raw preview: {raw[:300]}")

    # Clean markdown fences
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    # Extract JSON boundaries
    start = raw.find('{')
    end   = raw.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object in LLM response: {raw[:300]}")
    raw = raw[start:end]

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"⚠️ Direct JSON parse failed: {e}")

    # Attempt 2: extract file_path and explanation with regex,
    # rebuild fixed_content separately to avoid multiline issues
    try:
        file_path_m   = re.search(r'"file_path"\s*:\s*"([^"]+)"', raw)
        explanation_m = re.search(r'"explanation"\s*:\s*"([^"]+)"', raw)

        # Extract fixed_content between first and last quote of its value
        fc_start = raw.find('"fixed_content"')
        if fc_start == -1:
            raise ValueError("No fixed_content key found")
        colon_pos   = raw.find(':', fc_start)
        quote_open  = raw.find('"', colon_pos + 1)
        # Walk to find unescaped closing quote
        i = quote_open + 1
        while i < len(raw):
            if raw[i] == '\\':
                i += 2
                continue
            if raw[i] == '"':
                break
            i += 1
        fixed_raw = raw[quote_open + 1: i]
        fixed_raw = fixed_raw.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')

        if not file_path_m or not fixed_raw:
            raise ValueError("Could not extract required fields")

        result = {
            "file_path":     file_path_m.group(1),
            "fixed_content": fixed_raw,
            "explanation":   explanation_m.group(1) if explanation_m else "Fixed as requested"
        }
        print("✅ JSON recovered via regex fallback")
        return result

    except Exception as ex:
        raise ValueError(f"All JSON parse attempts failed. Error: {ex}\nRaw: {raw[:500]}")

# ── Step 4: Get main branch SHA ──────────────────────────────────────────
def get_main_sha():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/ref/heads/main"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["object"]["sha"]

# ── Step 5: Create branch ────────────────────────────────────────────────
def create_branch(branch_name, sha):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/refs"
    # Delete branch if it already exists
    del_url = f"https://api.github.com/repos/{GITHUB_REPO}/git/refs/heads/{branch_name}"
    requests.delete(del_url, headers=HEADERS, timeout=30)  # ignore errors

    r = requests.post(url, headers=HEADERS, json={
        "ref": f"refs/heads/{branch_name}",
        "sha": sha
    }, timeout=30)
    r.raise_for_status()
    print(f"✅ Branch created: {branch_name}")

# ── Step 6: Commit the fix ───────────────────────────────────────────────
def commit_fix(branch_name, file_path, new_content, explanation):
    _, file_sha = get_file(file_path, branch=branch_name)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    body = {
        "message": f"fix: resolve issue #{ISSUE_NUMBER} - {explanation[:60]}",
        "content": base64.b64encode(new_content.encode("utf-8")).decode(),
        "branch":  branch_name
    }
    if file_sha:
        body["sha"] = file_sha

    r = requests.put(url, headers=HEADERS, json=body, timeout=30)
    r.raise_for_status()
    print(f"✅ Committed fix to {file_path}")

# ── Step 7: Open a PR ────────────────────────────────────────────────────
def create_pr(branch_name, explanation):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls"
    r = requests.post(url, headers=HEADERS, json={
        "title": f"fix: resolve issue #{ISSUE_NUMBER}",
        "body":  f"Fixes #{ISSUE_NUMBER}\n\n**What changed:**\n{explanation}\n\n> 🤖 Auto-generated by Issue Agent",
        "head":  branch_name,
        "base":  "main",
        "draft": True
    }, timeout=30)
    r.raise_for_status()
    pr_url = r.json()["html_url"]
    print(f"✅ PR created: {pr_url}")
    return pr_url

# ── Step 8: Comment on issue ─────────────────────────────────────────────
def comment_on_issue(pr_url, explanation):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{ISSUE_NUMBER}/comments"
    r = requests.post(url, headers=HEADERS, json={
        "body": (
            f"🤖 **Issue Agent** has attempted a fix!\n\n"
            f"**Changes:** {explanation}\n\n"
            f"**Pull Request:** {pr_url}\n\n"
            f"Please review and merge if the fix looks correct."
        )
    }, timeout=30)
    r.raise_for_status()
    print("✅ Commented on issue")

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("🤖 Issue Agent Starting")
    print("=" * 50)

    # 1. Get issue
    title, body = get_issue()

    # 2. Read codebase
    py_files = get_repo_tree()
    files_content = ""
    for path in py_files:
        content, _ = get_file(path)
        if content:
            # Limit each file to 2000 chars to stay within token limit
            files_content += f"\n\n### FILE: {path}\n```python\n{content[:2000]}\n```"
    print(f"✅ Loaded {len(py_files)} files for context")

    # 3. Ask LLM
    print("🧠 Asking LLM for fix...")
    result      = ask_llm(title, body, files_content)
    file_path   = result["file_path"]
    fixed       = result["fixed_content"]
    explanation = result["explanation"]
    print(f"✅ LLM fix target: {file_path}")
    print(f"✅ Explanation: {explanation}")

    # 4. Validate file exists in repo
    existing_content, _ = get_file(file_path)
    if existing_content is None:
        print(f"⚠️ File {file_path} not found, defaulting to streamlit_app.py")
        file_path = "streamlit_app.py"

    # 5. Create branch
    branch_name = f"fix/issue-{ISSUE_NUMBER}"
    main_sha    = get_main_sha()
    create_branch(branch_name, main_sha)

    # 6. Commit fix
    commit_fix(branch_name, file_path, fixed, explanation)

    # 7. Open PR
    pr_url = create_pr(branch_name, explanation)

    # 8. Comment on issue
    comment_on_issue(pr_url, explanation)

    print("=" * 50)
    print("✅ Issue Agent completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()