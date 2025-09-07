# streamlit_app.py
# ChatGPT-style app powered by local Ollama models (offline).
# Run: streamlit run streamlit_app.py

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from PyPDF2 import PdfReader
import ollama

# ----------------------------
# Constants & Paths
# ----------------------------
APP_TITLE = "Offline ChatGPT (Ollama)"
BASE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SESSIONS_DIR = os.path.join(STORAGE_DIR, "sessions")
FILES_DIR = os.path.join(STORAGE_DIR, "files")

# Clarified system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "The user may upload files, and ask you to summarise it. "
    "You won't get the actual files, but their text content along with the prompt "
    "inside blocks that look like:\n\n"
    "FileName -> Begin Content\n"
    "<file text here>\n"
    "End Content\n\n"
    "Always use this embedded content as if you had opened the file yourself. "
    "Do NOT say 'I don’t see a file' — the file text is already given above. "
    "When answering, cite the filename when relevant."
)

MAX_FILE_CHARS = 200_000  # cap per-file text
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

# ----------------------------
# Utilities: sessions & storage
# ----------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def load_session(session_id: str) -> Dict[str, Any]:
    path = _session_path(session_id)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "created_at" not in data:
        data["created_at"] = datetime.fromtimestamp(os.path.getctime(path)).isoformat(timespec="seconds")
    if "updated_at" not in data:
        data["updated_at"] = data["created_at"]
    if "system" not in data:
        data["system"] = DEFAULT_SYSTEM_PROMPT
    if "files" not in data:
        data["files"] = []
    if "messages" not in data:
        data["messages"] = []
    if "temperature" not in data:
        data["temperature"] = 0.2
    return data

def save_session(data: Dict[str, Any]) -> None:
    data["updated_at"] = _now_iso()
    sid = data.get("id")
    with open(_session_path(sid), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def list_sessions() -> List[Dict[str, Any]]:
    items = []
    for name in sorted(os.listdir(SESSIONS_DIR)):
        if name.endswith(".json"):
            sid = name[:-5]
            try:
                data = load_session(sid)
                items.append(data)
            except Exception:
                continue
    items.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
    return items

def new_session(title: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    sid = str(uuid.uuid4())
    data = {
        "id": sid,
        "title": title or "New chat",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "model": model or "",
        "system": DEFAULT_SYSTEM_PROMPT,
        "messages": [],
        "files": [],
        "temperature": 0.2,
    }
    save_session(data)
    return data

def delete_session(session_id: str) -> None:
    try:
        os.remove(_session_path(session_id))
    except FileNotFoundError:
        pass

def rename_session(session_id: str, new_title: str) -> None:
    data = load_session(session_id)
    data["title"] = new_title.strip() or data["title"]
    save_session(data)

# ----------------------------
# Utilities: file handling & hashing
# ----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def compute_uploaded_hash(uploaded_file) -> str:
    try:
        mv = uploaded_file.getbuffer()
        b = mv.tobytes()
    except Exception:
        uploaded_file.seek(0)
        b = uploaded_file.read()
        uploaded_file.seek(0)
    return sha256_bytes(b)

def store_uploaded_file(uploaded_file) -> Dict[str, Any]:
    try:
        raw = uploaded_file.getbuffer().tobytes()
    except Exception:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        uploaded_file.seek(0)

    sha = sha256_bytes(raw)
    _, ext = os.path.splitext(uploaded_file.name)
    disk_name = f"{sha}{ext}"
    path = os.path.join(FILES_DIR, disk_name)

    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)

    return {
        "id": sha,
        "name": uploaded_file.name,
        "path": path,
        "size": len(raw),
        "hash": sha,
        "created_at": _now_iso(),
    }

def read_text_from_file(meta: Dict[str, Any]) -> str:
    name_lower = meta.get("name").lower()
    path = meta.get("path")
    try:
        if name_lower.endswith((".txt", ".md", ".py", ".log", ".json", ".yaml", ".yml", ".csv")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if name_lower.endswith(".pdf"):
            with open(path, "rb") as f:
                reader = PdfReader(f)
                return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        with open(path, "rb") as f:
            raw = f.read()
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.decode("latin-1", errors="ignore")
    except Exception as e:
        return f"[Error reading file '{meta.get('name')}': {e}]"

def format_files_for_context(file_metas: List[Dict[str, Any]]) -> str:
    blocks = []
    for fm in file_metas:
        text = read_text_from_file(fm)
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + f"\n[Truncated at {MAX_FILE_CHARS} characters]"
        block = f"{fm['name']} -> Begin Content\n{text}\nEnd Content"
        blocks.append(block)
    return "\n\n".join(blocks)

def remove_file_from_disk_if_unreferenced(file_hash: str):
    for sess in list_sessions():
        for fm in sess.get("files", []):
            if fm.get("hash") == file_hash or fm.get("id") == file_hash:
                return
    for fname in os.listdir(FILES_DIR):
        if fname.startswith(file_hash):
            try:
                os.remove(os.path.join(FILES_DIR, fname))
            except Exception:
                pass

# ----------------------------
# Utilities: Ollama chat
# ----------------------------
def list_ollama_models() -> List[str]:
    try:
        res = ollama.list()
        models = []
        for m in res.get("models", []):
            if isinstance(m, dict):
                nm = m.get("name") or m.get("model") or m.get("id")
            else:
                nm = getattr(m, "name", None) or getattr(m, "model", None)
            if nm:
                models.append(nm)
        return models
    except Exception as e:
        st.warning(f"Could not list Ollama models: {e}")
        return []

def stream_ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2):
    try:
        print(messages)
        stream = ollama.chat(model=model, messages=messages, stream=True, options={"temperature": temperature})
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
    except Exception as e:
        yield f"\n[Error from model: {e}]"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")

# initialize
if "loaded" not in st.session_state:
    st.session_state.loaded = True
    sessions = list_sessions()
    if not sessions:
        sess = new_session(title="Welcome")
    else:
        sess = sessions[0]
    st.session_state.current_session_id = sess["id"]

# Sidebar
with st.sidebar:
    st.title("💬 Chat History")

    models = list_ollama_models()
    if not models:
        st.error("No Ollama models found. Use `ollama pull <model>` to install one.")
    default_model = models[0] if models else ""

    sessions = list_sessions()
    current_id = st.session_state.get("current_session_id")
    current = next((s for s in sessions if s["id"] == current_id), sessions[0] if sessions else None)
    if current:
        st.session_state.current_session_id = current["id"]

    if st.button("➕ New chat", use_container_width=True):
        new = new_session(model=default_model)
        st.session_state.current_session_id = new["id"]
        st.rerun()

    search = st.text_input("🔍 Search chats")
    st.caption("Click a chat to open it. Use ⋯ for actions.")
    for s in [s for s in sessions if not search or search.lower() in s["title"].lower()]:
        cols = st.columns([0.8, 0.2])
        if cols[0].button(s["title"], key=f"open_{s['id']}", use_container_width=True):
            st.session_state.current_session_id = s["id"]
            st.rerun()
        with cols[1]:
            with st.popover("⋯"):
                new_title = st.text_input("Rename", value=s["title"], key=f"rename_{s['id']}")
                if st.button("Save name", key=f"save_rename_{s['id']}"):
                    rename_session(s["id"], new_title)
                    st.rerun()
                st.divider()
                if st.button("Delete", type="primary", key=f"delete_{s['id']}"):
                    delete_session(s["id"])
                    st.rerun()
                st.divider()
                data = json.dumps(s, ensure_ascii=False, indent=2)
                st.download_button("Export JSON", data=data, file_name=f"chat_{s['title']}.json")

    st.divider()
    st.subheader("Settings")
    if current:
        selected_model = st.selectbox("Model", models, index=models.index(current.get("model")) if current.get("model") in models else 0)
        temp = st.slider("Temperature", 0.0, 1.0, value=current.get("temperature", 0.2))
        sys_prompt = st.text_area("System Prompt", value=current.get("system", DEFAULT_SYSTEM_PROMPT))
        if st.button("Save Settings"):
            current["model"] = selected_model
            current["system"] = sys_prompt
            current["temperature"] = temp
            save_session(current)
            st.success("Saved")

        st.subheader("Import Chat")
        imported = st.file_uploader("Upload exported chat JSON", type="json")
        if imported:
            try:
                data = json.load(imported)
                if "id" not in data:
                    data["id"] = str(uuid.uuid4())
                save_session(data)
                st.success("Imported!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to import: {e}")

# ----------------------------
# Main area
# ----------------------------
sessions = list_sessions()
current_id = st.session_state.get("current_session_id")
if not current_id:
    st.stop()
try:
    current = load_session(current_id)
except Exception:
    st.error("Could not load the selected session. Create a new chat from the sidebar.")
    st.stop()

st.title(APP_TITLE)

# File uploader
with st.expander("📎 Upload files (txt, md, csv, pdf)", expanded=False):
    uploads = st.file_uploader("Add one or more files", type=["txt", "md", "csv", "pdf", "json", "yaml", "yml", "py", "log"], accept_multiple_files=True)
    if uploads:
        existing_hashes = {fm.get("hash") or fm.get("id") for fm in current.get("files", [])}
        new_files = []
        for uf in uploads:
            try:
                h = compute_uploaded_hash(uf)
                if h in existing_hashes:
                    continue
                meta = store_uploaded_file(uf)
                meta["hash"] = meta.get("hash") or meta.get("id")
                new_files.append(meta)
                existing_hashes.add(meta["hash"])
            except Exception as e:
                st.warning(f"Failed to add {getattr(uf, 'name', 'file')}: {e}")
        if new_files:
            current["files"].extend(new_files)
            save_session(current)
            st.success(f"Added {len(new_files)} new file(s)")
        else:
            st.info("No new files to add")

    if current.get("files"):
        st.caption("Files in this chat:")
        for fm in list(current.get("files", [])):
            fc1, fc2 = st.columns([0.85, 0.15])
            fc1.write(f"• {fm['name']} ({fm.get('size', '?')} bytes)")
            if fc2.button("Remove", key=f"rm_{fm['id']}"):
                file_hash = fm.get("hash") or fm.get("id")
                current["files"] = [x for x in current["files"] if (x.get("hash") or x.get("id")) != file_hash]
                save_session(current)
                remove_file_from_disk_if_unreferenced(file_hash)
                st.rerun()

# Chat history
for m in current.get("messages", []):
    if m.get("role") == "system":
        continue
    with st.chat_message(m.get("role")):
        st.markdown(m.get("content"))

# Compose area
user_prompt = st.chat_input("Message…")
if user_prompt:
    convo = [{
        "role": "system", 
        "content": current.get("system", DEFAULT_SYSTEM_PROMPT)
        }]
    user_msg = {"role": "user", "content": user_prompt}
    
    if current.get("files"):
        file_block = format_files_for_context(current.get("files"))
        user_msg["content"] += "\n" + file_block    
    convo.extend([m for m in current.get("messages", []) if m.get("role") != "system"])

    current["messages"].append(user_msg)
    if current["title"] in ["New chat", "Welcome"]:
        current["title"] = user_prompt[:40]
    save_session(current)

    with st.chat_message("user"):
        st.markdown(user_prompt.split("\n")[0])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        accum = ""
        model_to_use = current.get("model") or (list_ollama_models()[:1] or [""])[0]
        for token in stream_ollama_chat(model=model_to_use, messages=convo + [user_msg], temperature=current.get("temperature", 0.2)):
            accum += token
            placeholder.markdown(accum)
        assistant_msg = {"role": "assistant", "content": accum}
        current["messages"].append(assistant_msg)
        save_session(current)

# Footer
st.divider()
cols = st.columns([0.2, 0.2, 0.6])
if cols[0].button("🧹 Clear chat (keep files)"):
    current["messages"] = []
    save_session(current)
    st.rerun()

if cols[1].button("🗑️ Clear files (keep chat)"):
    hashes_to_remove = [(fm.get("hash") or fm.get("id")) for fm in current.get("files", [])]
    current["files"] = []
    save_session(current)
    for h in hashes_to_remove:
        remove_file_from_disk_if_unreferenced(h)
    st.rerun()

st.markdown(
    """
    <style>
    .stChatMessage { font-size: 0.98rem; }
    .stButton button { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)
