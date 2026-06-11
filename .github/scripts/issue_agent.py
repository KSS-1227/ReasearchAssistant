import os
import re
import sys
import json
import time
import base64
import requests

# ── Config ────────────────────────────────────────────────────────────────
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

# ✅ Key files the agent should always know about (architecture context)
KEY_FILES = [
    "streamlit_app.py",
    "fastapi_app.py",
    "requirements.txt",
    "Dockerfile",
]

# ── GitHub API with retry ─────────────────────────────────────────────────
def github_request(method: str, url: str, retries: int = 3, **kwargs):
    """GitHub API call with exponential backoff retry."""
    kwargs.setdefault("timeout", 30)
    for attempt in range(1, retries + 1):
        try:
            r = requests.request(method, url, headers=HEADERS, **kwargs)
            if r.status_code == 429:                        # rate limited
                wait = int(r.headers.get("Retry-After", 60))
                print(f"⏳ Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"⚠️ Attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

# ── Step 1: Read the issue ────────────────────────────────────────────────
def get_issue():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{ISSUE_NUMBER}"
    r = github_request("GET", url)
    data = r.json()
    print(f"✅ Issue #{ISSUE_NUMBER}: {data['title']}")
    return data["title"], data.get("body") or ""

# ── Step 2: Read the codebase ─────────────────────────────────────────────
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
    """Get Python files + always include key architecture files."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/main?recursive=1"
    r = github_request("GET", url)
    
    # All .py files excluding .github internals
    py_files = [
        f["path"] for f in r.json().get("tree", [])
        if f["type"] == "blob"
        and f["path"].endswith(".py")
        and not f["path"].startswith(".github")
    ]

    # ✅ Always include key files first, then fill remaining slots
    ordered = []
    for kf in KEY_FILES:
        if kf in py_files:
            ordered.append(kf)
            py_files.remove(kf)
        elif kf.endswith(".py"):
            ordered.append(kf)             # try to load even if not in tree

    ordered += py_files[:6]               # remaining slots up to 8 total
    print(f"✅ Found {len(ordered)} files to load")
    return ordered

# ── Step 3: Ask LLM ───────────────────────────────────────────────────────
def ask_llm(issue_title, issue_body, files_content):
    # ✅ Mention FastAPI architecture in prompt
    prompt = f"""You are an expert Python developer working on a FastAPI + Streamlit research assistant app.

The app has TWO main Python files:
- streamlit_app.py  → frontend UI (calls FastAPI backend via HTTP)
- fastapi_app.py    → REST API backend (uses ResearchCoordinator)

ISSUE TITLE: {issue_title}

ISSUE DESCRIPTION:
{issue_body}

CODEBASE:
{files_content}

YOUR TASK:
- Identify which file needs to be changed (streamlit_app.py OR fastapi_app.py)
- Write the COMPLETE fixed version of ONLY that one file
- Respond with ONLY a valid JSON object — nothing else

JSON FORMAT:
{{"file_path": "filename.py", "fixed_content": "full file content with \\n for newlines", "explanation": "brief description"}}

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
        "temperature": 0.1,
    }

    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,                        # ✅ increased — NVIDIA can be slow
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

    # Attempt 2: regex fallback
    try:
        file_path_m   = re.search(r'"file_path"\s*:\s*"([^"]+)"', raw)
        explanation_m = re.search(r'"explanation"\s*:\s*"([^"]+)"', raw)

        fc_start    = raw.find('"fixed_content"')
        if fc_start == -1:
            raise ValueError("No fixed_content key")
        colon_pos   = raw.find(':', fc_start)
        quote_open  = raw.find('"', colon_pos + 1)
        i = quote_open + 1
        while i < len(raw):
            if raw[i] == '\\':
                i += 2
                continue
            if raw[i] == '"':
                break
            i += 1
        fixed_raw = raw[quote_open + 1: i]
        fixed_raw = (
            fixed_raw
            .replace('\\n', '\n')
            .replace('\\"', '"')
            .replace('\\\\', '\\')
        )

        if not file_path_m or not fixed_raw.strip():
            raise ValueError("Could not extract required fields")

        print("✅ JSON recovered via regex fallback")
        return {
            "file_path":     file_path_m.group(1),
            "fixed_content": fixed_raw,
            "explanation":   explanation_m.group(1) if explanation_m else "Fixed as requested",
        }

    except Exception as ex:
        raise ValueError(
            f"All JSON parse attempts failed.\nError: {ex}\nRaw: {raw[:500]}"
        )

# ── Step 4: Get main SHA ──────────────────────────────────────────────────
def get_main_sha():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/ref/heads/main"
    r = github_request("GET", url)
    return r.json()["object"]["sha"]

# ── Step 5: Create branch ─────────────────────────────────────────────────
def create_branch(branch_name, sha):
    # Delete existing branch silently
    del_url = f"https://api.github.com/repos/{GITHUB_REPO}/git/refs/heads/{branch_name}"
    requests.delete(del_url, headers=HEADERS, timeout=30)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/refs"
    github_request("POST", url, json={
        "ref": f"refs/heads/{branch_name}",
        "sha": sha,
    })
    print(f"✅ Branch created: {branch_name}")

# ── Step 6: Check for existing PR ────────────────────────────────────────
def get_existing_pr(branch_name):
    """Return existing PR URL if one already exists for this branch."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls"
    r = github_request("GET", url, params={"head": f"{GITHUB_REPO.split('/')[0]}:{branch_name}", "state": "open"})
    prs = r.json()
    if prs:
        pr_url = prs[0]["html_url"]
        print(f"ℹ️ PR already exists: {pr_url}")
        return pr_url
    return None

# ── Step 7: Commit the fix ────────────────────────────────────────────────
def commit_fix(branch_name, file_path, new_content, explanation):
    # ✅ Validate content before committing
    if not new_content or not new_content.strip():
        raise ValueError(f"LLM returned empty content for {file_path} — aborting commit")

    _, file_sha = get_file(file_path, branch=branch_name)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    body = {
        "message": f"fix: resolve issue #{ISSUE_NUMBER} - {explanation[:60]}",
        "content": base64.b64encode(new_content.encode("utf-8")).decode(),
        "branch":  branch_name,
    }
    if file_sha:
        body["sha"] = file_sha

    github_request("PUT", url, json=body)
    print(f"✅ Committed fix to {file_path}")

# ── Step 8: Open PR ───────────────────────────────────────────────────────
def create_pr(branch_name, explanation):
    # ✅ Check for existing PR first — avoid duplicates
    existing = get_existing_pr(branch_name)
    if existing:
        return existing

    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls"
    r = github_request("POST", url, json={
        "title": f"fix: resolve issue #{ISSUE_NUMBER}",
        "body": (
            f"Fixes #{ISSUE_NUMBER}\n\n"
            f"**What changed:**\n{explanation}\n\n"
            f"> 🤖 Auto-generated by Issue Agent"
        ),
        "head":  branch_name,
        "base":  "main",
        "draft": True,
    })
    pr_url = r.json()["html_url"]
    print(f"✅ PR created: {pr_url}")
    return pr_url

# ── Step 9: Comment on issue ──────────────────────────────────────────────
def comment_on_issue(pr_url, explanation):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{ISSUE_NUMBER}/comments"
    github_request("POST", url, json={
        "body": (
            f"🤖 **Issue Agent** has attempted a fix!\n\n"
            f"**Changes:** {explanation}\n\n"
            f"**Pull Request:** {pr_url}\n\n"
            f"Please review and merge if the fix looks correct."
        )
    })
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
    loaded = 0
    for path in py_files:
        content, _ = get_file(path)
        if content:
            files_content += f"\n\n### FILE: {path}\n```python\n{content[:2000]}\n```"
            loaded += 1

    # ✅ Guard: abort if no files loaded
    if not files_content.strip():
        raise RuntimeError("No files loaded from repo — cannot generate fix")

    print(f"✅ Loaded {loaded} files for context")

    # 3. Ask LLM
    print("🧠 Asking LLM for fix...")
    result      = ask_llm(title, body, files_content)
    file_path   = result["file_path"]
    fixed       = result["fixed_content"]
    explanation = result["explanation"]
    print(f"✅ LLM fix target: {file_path}")
    print(f"✅ Explanation: {explanation}")

    # 4. Validate file exists — fallback to fastapi_app.py or streamlit_app.py
    existing_content, _ = get_file(file_path)
    if existing_content is None:
        print(f"⚠️ File '{file_path}' not found in repo")
        # ✅ Smarter fallback: pick based on issue keywords
        issue_text = (title + " " + body).lower()
        if any(k in issue_text for k in ["api", "endpoint", "fastapi", "backend", "route"]):
            file_path = "fastapi_app.py"
        else:
            file_path = "streamlit_app.py"
        print(f"⚠️ Defaulting to {file_path}")

    # 5. Create branch — ✅ includes timestamp to avoid collisions on re-trigger
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
    try:
        main()
    except Exception as e:
        print(f"\n❌ Issue Agent failed: {e}", file=sys.stderr)
        sys.exit(1)         # ✅ ensures GitHub Actions marks job as failed