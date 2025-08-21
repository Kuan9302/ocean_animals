import streamlit as st
import requests
from PIL import Image
import io
from pathlib import Path
import base64, mimetypes
import streamlit.components.v1 as components
import re

# ---------- utilities ----------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

st.set_page_config(page_title="海洋生物辨識系統", page_icon="🌊", layout="wide")
API_BASE = "http://127.0.0.1:8000"  # FastAPI

# === 背景圖片 ===
BG_IMG = Path(r"C:\Users\user\Desktop\ocean_animals\ocean.jpg")

def set_background(img_path: Path):
    if not img_path.exists():
        st.error(f"找不到圖片：{img_path}")
        return
    mime = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:{mime};base64,{b64}");
            background-size: cover; background-position: center;
            background-attachment: fixed; background-repeat: no-repeat;
        }}
        .block-container {{
            background: rgba(255,255,255,.78);
            border-radius: 16px; padding: 1rem 1.25rem; padding-top: 64px; margin-top: 8px;
        }}
        .stApp, .stApp * {{ color:#000 !important; }}
        header[data-testid="stHeader"]{{background:transparent!important}}
        div[data-testid="stToolbar"]{{
            background: rgba(255,255,255,.98) !important;
            border-bottom: 1px solid rgba(0,0,0,.08);
            z-index: 10000;
        }}
        div[data-testid="stToolbar"] * {{ color:#000!important; fill:#000!important; }}

        section[data-testid="stSidebar"]{{ z-index:1000; }}
        section[data-testid="stSidebar"]>div{{
            background: rgba(255,255,255,.90); border-radius: 12px;
            margin-top: 64px; padding: 12px; box-shadow: 0 4px 12px rgba(0,0,0,.08);
        }}

        .stButton>button, button,
        .stDownloadButton>button,
        div[data-testid="stFormSubmitButton"] button,
        section[data-testid="stSidebar"] .stButton>button,
        div[data-testid="stFileUploader"] button {{
            background:#fff !important; color:#000 !important;
            border:1.5px solid #000 !important; border-radius:10px !important; box-shadow:none !important;
            display:inline-flex; align-items:center; justify-content:center;
            height:40px; padding:0 16px; line-height:40px;
        }}
        section[data-testid="stSidebar"] .stButton>button {{ width:100%; font-weight:600; }}
        .stButton>button:hover, button:hover {{ filter: brightness(1.02); }}
        .stButton>button:active, button:active {{ filter: brightness(.98); }}

        header [data-testid="baseButton-header"],
        header [data-testid="baseButton-headerNoPadding"],
        header [data-testid="collapsedControl"] button {{
            background:#fff!important; color:#000!important;
            border:1.5px solid #000!important; border-radius:10px!important; box-shadow:none!important;
        }}

        div[data-baseweb="input"],
        div[data-baseweb="select"],
        div[data-baseweb="textarea"] {{
            background:#fff !important; border:1.5px solid #000 !important;
            border-radius:10px !important; box-shadow:none !important;
        }}
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {{
            background:#fff !important; color:#000 !important;
        }}
        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {{ color:#333 !important; opacity:.85; }}

        div[data-testid="stFileUploaderDropzone"] {{
            background:#fff !important; color:#000 !important;
            border:1.5px solid #000 !important; border-radius:10px !important;
            box-shadow:none !important;
        }}
        div[data-testid="stFileUploaderDropzone"] * {{ color:#000 !important; }}
        div[data-testid="stFileUploader"] span {{ color:#000 !important; }}

        section[data-testid="stSidebar"] div[role="tablist"] {{ gap: 10px; }}
        section[data-testid="stSidebar"] button[role="tab"] {{
            background:#fff !important; color:#000 !important;
            border:1.5px solid #000 !important; border-radius:10px !important;
            height:38px; min-width:100px; padding:0 18px;
            display:inline-flex; align-items:center; justify-content:center;
            box-shadow:none !important;
        }}
        section[data-testid="stSidebar"] button[role="tab"]:hover {{ filter: brightness(1.02); }}
        div[role="tablist"]{{ margin-top:8px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background(BG_IMG)
st.markdown("<h1 style='text-align:center;margin-top:8px;color:#000'>海洋生物辨識系統</h1>", unsafe_allow_html=True)

components.html("""
<script>
(function(){
  const kill = () => {
    // 關閉所有表單的自動填入
    document.querySelectorAll('form').forEach(f => f.setAttribute('autocomplete','off'));

    // 讓所有密碼欄位用 new-password，並先 readonly 避免被偵測成可自動填入
    document.querySelectorAll('input[type="password"]').forEach(el => {
      el.setAttribute('autocomplete','new-password');
      if (!el.hasAttribute('data-pm-guard')) {
        el.setAttribute('data-pm-guard','1');
        el.setAttribute('readonly','readonly');
        el.addEventListener('focus', () => el.removeAttribute('readonly'), { once:true });
      }
    });

    // 常見的帳號/Email欄位也關閉自動填入
    document.querySelectorAll('input[type="text"],input[type="email"]').forEach(el=>{
      const hint = (el.getAttribute('aria-label')||el.name||'').toLowerCase();
      if (/(user|account|email|mail|帳|號|用戶|使用者|name)/.test(hint)) {
        el.setAttribute('autocomplete','off');
      }
    });
  };
  // 初次執行 + 之後 DOM 更新也自動套用
  new MutationObserver(kill).observe(document.documentElement,{childList:true,subtree:true});
  kill();
})();
</script>
""", height=0, width=0)

# ======== 狀態 ========
if "token" not in st.session_state: st.session_state.token = None
if "username" not in st.session_state: st.session_state.username = None
if "last_result" not in st.session_state: st.session_state.last_result = None
if "chat_cache" not in st.session_state: st.session_state.chat_cache = {}

# ---------- auth sidebar ----------
with st.sidebar:
    st.header("帳號")
    if st.session_state.get("token"):
        st.success(f"已登入：{st.session_state.get('username')}")
        try:
            h = requests.get(f"{API_BASE}/ollama/health", timeout=3)
            if h.ok and h.headers.get("content-type","").startswith("application/json"):
                ok = h.json().get("ok"); model = h.json().get("model")
                st.caption(("✅ Ollama 正常" if ok else "⚠️ Ollama 未就緒") + f"（model: {model}）")
        except Exception:
            pass

        def _logout():
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.last_result = None
            st.session_state.chat_cache = {}
            safe_rerun()

        st.button("登出", key="btn_logout", on_click=_logout)
    else:
        tab_reg, tab_login = st.tabs(["註冊", "登入"])
        with tab_reg:
            r_user = st.text_input("新帳號", key="reg_user")
            r_pwd  = st.text_input("新密碼", type="password", key="reg_pwd")
            if st.button("註冊", key="btn_register"):
                if not r_user or not r_pwd:
                    st.warning("請輸入帳號與密碼")
                else:
                    resp = requests.post(f"{API_BASE}/register", json={"username": r_user, "password": r_pwd})
                    if resp.status_code == 201:
                        st.success("註冊成功，請切到『登入』")
                    else:
                        try:
                            st.error(resp.json().get("detail") or f"註冊失敗 ({resp.status_code})")
                        except Exception:
                            st.error(f"註冊失敗 ({resp.status_code})")
        with tab_login:
            l_user = st.text_input("帳號", key="login_user")
            l_pwd  = st.text_input("密碼", type="password", key="login_pwd")
            if st.button("登入", key="btn_login"):
                data = {"username": l_user, "password": l_pwd}
                resp = requests.post(f"{API_BASE}/login", data=data)
                if resp.status_code == 200:
                    st.session_state.token = resp.json()["access_token"]
                    st.session_state.username = l_user
                    st.success("登入成功！")
                    safe_rerun()
                else:
                    try:
                        st.error(resp.json().get("detail") or f"登入失敗 ({resp.status_code})")
                    except Exception:
                        st.error(f"登入失敗 ({resp.status_code})")

# ---------- helper: render species info ----------
def render_bio_info_from_any(data: dict):
    kb = data.get("kb_info") or data.get("info_kb") or {}
    llm = data.get("llm_info") or data.get("info_llm") or {}

    def pick(k):
        return (kb.get(k) if isinstance(kb, dict) else None) or (llm.get(k) if isinstance(llm, dict) else None)

    # 多了「行為」
    rows = ["中文名稱", "英文名稱", "分類", "常見棲地與分布", "生物型態", "行為"]

    def oneline(key: str, val):
        # 「生物型態」與「行為」都壓成單行：把換行轉成全形分號
        if isinstance(val, str) and key in {"生物型態", "行為"}:
            val = re.sub(r'\s*[\r\n]+\s*', '', val)
            val = re.sub(r'；{2,}', '；', val).strip('').strip()
        return val

    has_any = any(pick(k) for k in rows)
    if has_any:
        st.subheader("生物資訊")
        for k in rows:
            v = oneline(k, pick(k))
            if v:
                st.markdown(f"**{k}：** {v}")
    else:
        st.caption("此筆尚無介紹。")



def _subject_from(data: dict) -> str | None:
    kb = data.get("kb_info") or data.get("info_kb") or {}
    llm = data.get("llm_info") or data.get("info_llm") or {}
    top = (data.get("top_prediction") or {}).get("class_name") or data.get("top_class")
    return kb.get("中文名稱") or llm.get("中文名稱") or top

# ---------- chat box ----------
def render_chat_box(record_id: int, chat_list: list, token: str, scope: str = "main",
                    subject_name: str | None = None):
    st.subheader("與海洋助理聊天")
    if subject_name:
        st.caption(f"本次對話以「{subject_name}」為主題（已鎖定）。")

    # 顯示最近對話
    if chat_list:
        for m in chat_list[-10:]:
            who = "🧑‍💻 使用者" if m.get("role") == "user" else "🤖 助理"
            st.markdown(f"**{who}：** {m.get('content','')}")

    form_key  = f"chat_form_{scope}_{record_id}"
    input_key = f"chat_q_{scope}_{record_id}"

    with st.form(form_key, clear_on_submit=True):
        q = st.text_input(f"輸入你的問題（record {record_id}）", key=input_key)
        ok_submit = st.form_submit_button("送出問題")

    if ok_submit:
        if not q.strip():
            st.warning("請先輸入問題再送出。")
            return
        headers = {"Authorization": f"Bearer {token}"}
        with st.spinner("助理思考中..."):
            r = requests.post(
                f"{API_BASE}/chat",
                headers=headers,
                json={"record_id": record_id, "question": q, "free": False},  # 鎖定物種
                timeout=120
            )
        if r.status_code == 200:
            ans = (r.json() or {}).get("answer") or "（沒有產生內容）"
            st.success("已回覆：")
            st.write(ans)
            # 即時寫入 chat_cache 並刷新畫面
            cache = st.session_state.chat_cache.get(record_id, [])[:]
            now_pair = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans},
            ]
            cache.extend(now_pair)
            st.session_state.chat_cache[record_id] = cache
            safe_rerun()
        else:
            try:
                st.error(r.json().get("detail") or f"送出失敗（{r.status_code}）")
            except Exception:
                st.error(f"送出失敗（{r.status_code}）")

# ---------- tabs ----------
tab_upload, tab_history = st.tabs(["上傳辨識", "歷史紀錄"])

with tab_upload:
    if not st.session_state.token:
        st.warning("請先登入才能上傳圖片。")
    else:
        # 若有上次結果，先顯示它（可繼續聊天）
        if st.session_state.last_result:
            with st.container(border=True):
                r = st.session_state.last_result
                st.info(f"這是你上傳的第 {r.get('seq')} 張")
                top = r.get("top_prediction")
                if top:
                    st.success(f"最可能：{top['class_name']}（置信度 {top['confidence']:.8f}）")
                if r.get("image_url"):
                    st.image(f"{API_BASE}{r['image_url']}", use_container_width=True)
                render_bio_info_from_any(r)
                rid = r.get("record_id")
                if rid is not None:
                    subject = _subject_from(r)
                    cached = st.session_state.chat_cache.get(rid, r.get("chat", []))
                    render_chat_box(rid, cached, st.session_state.token, scope="upload", subject_name=subject)

            def _clear_last():
                st.session_state.last_result = None
                safe_rerun()
            st.button("清除目前結果", on_click=_clear_last, key="clear_last")

        st.divider()
        use_ollama = st.checkbox("使用 Ollama 產生生物解說", value=True)
        uploaded_file = st.file_uploader("請上傳海洋生物圖片", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None and st.button("開始辨識", key="btn_predict"):
            img_bytes = uploaded_file.getvalue()
            files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type or "image/jpeg")}
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            with st.spinner("辨識與產生介紹中..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/predict",
                        params={"explain": str(use_ollama).lower()},
                        files=files, headers=headers, timeout=120
                    )
                except requests.RequestException as e:
                    st.error(f"無法連線到後端：{e}")
                else:
                    if resp.status_code == 200:
                        st.session_state.last_result = resp.json()
                        rid = st.session_state.last_result.get("record_id")
                        if rid is not None:
                            st.session_state.chat_cache[rid] = st.session_state.last_result.get("chat", [])
                        safe_rerun()
                    else:
                        try:
                            st.error(resp.json().get("detail") or resp.json().get("error") or f"錯誤 {resp.status_code}")
                        except Exception:
                            st.error(f"錯誤 {resp.status_code}")

with tab_history:
    if not st.session_state.token:
        st.info("登入後可查看歷史紀錄。")
    else:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        limit = st.number_input("每次載入筆數", min_value=1, max_value=200, value=50, step=1)
        offset = st.number_input("從第幾筆開始 (offset)", min_value=0, value=0, step=1)

        if st.button("載入歷史", key="btn_history"):
            with st.spinner("讀取歷史中..."):
                resp = requests.get(f"{API_BASE}/history", headers=headers,
                                    params={"limit": int(limit), "offset": int(offset)})
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if not items:
                    st.write("尚無資料或已到底。")
                for it in items:
                    with st.container(border=True):
                        seq = it.get("seq")
                        st.write(f"#{seq}（第 {seq} 張）｜時間：{it.get('created_at')}")
                        st.write(f"Top：{it.get('top_class')}（{it.get('top_conf')}）")
                        if it.get("image_url"):
                            st.image(f"{API_BASE}{it['image_url']}", use_container_width=True)

                        with st.expander("查看 Top-K / 完整分佈"):
                            st.write("Top-K：", it.get("topk"))

                        render_bio_info_from_any(it)

                        # 若沒有介紹，提供一鍵補產生
                        if not (it.get("info_kb") or it.get("info_llm")):
                            if st.button("產生解說", key=f"explain_{it['id']}"):
                                with st.spinner("產生介紹中..."):
                                    try:
                                        r2 = requests.post(f"{API_BASE}/history/{it['id']}/explain",
                                                           headers=headers, timeout=120)
                                    except requests.RequestException as e:
                                        st.error(f"無法呼叫後端：{e}")
                                    else:
                                        if r2.status_code == 200:
                                            st.success("已產生介紹：")
                                            render_bio_info_from_any(r2.json())
                                        else:
                                            try:
                                                st.error(r2.json().get("detail") or f"產生失敗（{r2.status_code}）")
                                            except Exception:
                                                st.error(f"產生失敗（{r2.status_code}）")

                        # 聊天（帶入快取或既有）
                        rid = it["id"]
                        subject = _subject_from(it)
                        cached = st.session_state.chat_cache.get(rid, it.get("chat", []))
                        render_chat_box(rid, cached, st.session_state.token,
                                        scope=f"hist_{rid}", subject_name=subject)
            else:
                try:
                    st.error(resp.json().get("detail") or f"錯誤 {resp.status_code}")
                except Exception:
                    st.error(f"錯誤 {resp.status_code}")
