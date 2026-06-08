import os
import re
import json
import requests

# ── Config from environment ──────────────────────────────────────────────
GITHUB_TOKEN    = os.environ["GITHUB_TOKEN"]
GITHUB_REPO     = os.environ["GITHUB_REPO"]          # e.g. KSS-1227/ReasearchAssistant
ISSUE_NUMBER    = int(os.environ["ISSUE_NUMBER"])
LLM_API_KEY     = os.environ["LLM_API_KEY"]
LLM_BASE_URL    = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL       = os.environ.get("LLM_MODEL", "meta/llama-3.3-70b-instruct")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ── Step 1: Read the issue ───────────────────────────────────────────────
def get_issue():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{ISSUE_NUMBER}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    print(f"Issue #{ISSUE_NUMBER}: {data['title']}")
    return data["title"], data["body"] or ""

# ── Step 2: Read the codebase ────────────────────────────────────────────
def get_file(path):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    import base64
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]

def get_repo_tree():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/main?recursive=1"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    files = [f["path"] for f in r.json()["tree"]
             if f["type"] == "blob" and f["path"].endswith(".py")]
    return files[:10]  # limit to 10 py files

# ── Step 3: Ask LLM to fix the issue ────────────────────────────────────
def ask_llm(issue_title, issue_body, files_content):
    prompt = f"""You are an expert Python developer. Fix the following GitHub issue.

ISSUE TITLE: {issue_title}

ISSUE DESCRIPTION:
{issue_body}

CODEBASE:
{files_content}

YOUR TASK:
1. Identify which file needs to be changed
2. Write the complete fixed version of that file
3. Respond in this exact JSON format:
{{
  "file_path": "path/to/file.py",
  "fixed_content": "complete fixed file content here",
  "explanation": "what you changed and why"
}}

Respond with JSON only. No markdown, no explanation outside JSON."""

    response = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.1
        }
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    print(f"LLM response received ({len(raw)} chars)")

    # strip markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    # Fix unescaped control characters in JSON
    raw = raw.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON block manually
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

# ── Step 4: Create a branch ──────────────────────────────────────────────
def get_main_sha():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/ref/heads/main"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()["object"]["sha"]

def create_branch(branch_name, sha):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/refs"
    r = requests.post(url, headers=HEADERS, json={
        "ref": f"refs/heads/{branch_name}",
        "sha": sha
    })
    r.raise_for_status()
    print(f"Branch created: {branch_name}")

# ── Step 5: Commit the fix ───────────────────────────────────────────────
def commit_fix(branch_name, file_path, new_content, explanation):
    import base64
    _, file_sha = get_file(file_path)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    body = {
        "message": f"fix: resolve issue #{ISSUE_NUMBER} - {explanation[:60]}",
        "content": base64.b64encode(new_content.encode()).decode(),
        "branch": branch_name
    }
    if file_sha:
        body["sha"] = file_sha

    r = requests.put(url, headers=HEADERS, json=body)
    r.raise_for_status()
    print(f"Committed fix to {file_path}")

# ── Step 6: Open a PR ────────────────────────────────────────────────────
def create_pr(branch_name, explanation):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls"
    r = requests.post(url, headers=HEADERS, json={
        "title": f"fix: resolve issue #{ISSUE_NUMBER}",
        "body": f"Fixes #{ISSUE_NUMBER}\n\n**What changed:**\n{explanation}",
        "head": branch_name,
        "base": "main",
        "draft": True
    })
    r.raise_for_status()
    pr_url = r.json()["html_url"]
    print(f"PR created: {pr_url}")
    return pr_url

# ── Step 7: Comment on issue ─────────────────────────────────────────────
def comment_on_issue(pr_url, explanation):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{ISSUE_NUMBER}/comments"
    r = requests.post(url, headers=HEADERS, json={
        "body": f"🤖 **OpenHands Agent** has attempted a fix!\n\n**Changes made:** {explanation}\n\n**Pull Request:** {pr_url}\n\nPlease review the PR and merge if the fix looks correct."
    })
    r.raise_for_status()
    print("Commented on issue")

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=== Issue Agent Starting ===")

    # 1. Get issue
    title, body = get_issue()

    # 2. Read codebase
    py_files = get_repo_tree()
    files_content = ""
    for path in py_files:
        content, _ = get_file(path)
        if content:
            files_content += f"\n\n### {path}\n```python\n{content[:3000]}\n```"

    # 3. Ask LLM
    print("Asking LLM for fix...")
    result = ask_llm(title, body, files_content)
    file_path   = result["file_path"]
    fixed       = result["fixed_content"]
    explanation = result["explanation"]
    print(f"LLM wants to fix: {file_path}")
    print(f"Explanation: {explanation}")

    # 4. Create branch
    branch_name = f"fix/issue-{ISSUE_NUMBER}"
    main_sha = get_main_sha()
    create_branch(branch_name, main_sha)

    # 5. Commit fix
    commit_fix(branch_name, file_path, fixed, explanation)

    # 6. Open PR
    pr_url = create_pr(branch_name, explanation)

    # 7. Comment on issue
    comment_on_issue(pr_url, explanation)

    print("=== Agent completed successfully ===")

if __name__ == "__main__":
    main()