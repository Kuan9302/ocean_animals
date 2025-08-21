# -*- coding: utf-8 -*-

# ========= bcrypt 相容性（passlib 與 bcrypt 4.x）=========
import bcrypt as _bcrypt
if not hasattr(_bcrypt, "__about__"):
    class _About:
        __version__ = getattr(_bcrypt, "__version__", "4")
    _bcrypt.__about__ = _About()

import logging
logging.getLogger("passlib.handlers.bcrypt").setLevel(logging.ERROR)

# ========= 標準/第三方 =========
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, create_engine, func, Index, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from sqlalchemy.exc import IntegrityError, OperationalError

from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import io, json
from uuid import uuid4

import requests as http   # Ollama

# ========= App 設定 =========
app = FastAPI(title="Ocean AI API ...")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= JWT =========
SECRET_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET"   # ← 務必改成長隨機字串
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# ========= MySQL =========
DB_URL = "mysql+pymysql://root:ocean@localhost:3306/ocean?charset=utf8mb4"
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

print("=== Database Connection Info ===")
print("DB_URL in use:", engine.url)
try:
    with engine.connect() as conn:
        db_name = conn.execute(text("SELECT DATABASE()")).scalar()
        db_version = conn.execute(text("SELECT VERSION()")).scalar()
        print(f"Connected to database: {db_name}")
        print(f"MySQL version: {db_version}")
except Exception as e:
    print("❌ 無法連線到資料庫:", e)
print("================================")

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ========= 檔案上傳根目錄 & 靜態掛載 =========
BASE_UPLOAD_DIR = Path(r"C:\Users\user\ocean\uploads")
BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(BASE_UPLOAD_DIR)), name="files")

# ========= 資料庫模型 =========
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)

    # 每位使用者自己的第幾張（1 起跳）
    seq = Column(Integer, nullable=False)

    image_rel_path = Column(String(512), nullable=False)
    image_url = Column(String(512), nullable=False)
    top_class = Column(String(64), nullable=False)
    top_conf = Column(String(32), nullable=False)
    topk = Column(MySQLJSON, nullable=False)
    all_predictions = Column(MySQLJSON, nullable=False)

    # 介紹資料與對話
    info_kb  = Column(MySQLJSON, nullable=True)    # 來自 ocean.animal 的整理
    info_llm = Column(MySQLJSON, nullable=True)    # Ollama 產生
    chat     = Column(MySQLJSON, nullable=True)    # [{"role":"user/assistant","content":"...","ts":"..."}]

    created_at = Column(DateTime, server_default=func.now(), index=True)
    user = relationship("User", back_populates="predictions")

Index("ix_predictions_user_created", Prediction.user_id, Prediction.created_at.desc())
Base.metadata.create_all(bind=engine)

# ========= 啟動時自我修表 =========
def _ensure_column(conn, table: str, column: str, ddl: str):
    exist = conn.execute(text("""
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND COLUMN_NAME = :c
    """), {"t": table, "c": column}).scalar()
    if not exist:
        print(f"→ Adding column {table}.{column} ...")
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))

def ensure_schema():
    with engine.begin() as conn:
        # seq 欄位
        _ensure_column(conn, "predictions", "seq", "INT NOT NULL AFTER user_id")
        # 回填 seq
        conn.execute(text("SET SQL_SAFE_UPDATES=0"))
        conn.execute(text("""
            UPDATE predictions p
            JOIN (
              SELECT id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at, id) AS rn
              FROM predictions
            ) x ON x.id = p.id
            SET p.seq = x.rn
        """))
        conn.execute(text("SET SQL_SAFE_UPDATES=1"))
        # 唯一索引 (user_id, seq)
        uniq_exists = conn.execute(text("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'predictions'
              AND INDEX_NAME = 'ux_predictions_user_seq'
        """)).scalar()
        if not uniq_exists:
            print("→ Creating unique index ux_predictions_user_seq ...")
            try:
                conn.execute(text("CREATE UNIQUE INDEX ux_predictions_user_seq ON predictions(user_id, seq)"))
            except Exception as e:
                print("⚠️ 建立唯一索引時出錯（可能已存在）：", e)
        # JSON 欄位
        _ensure_column(conn, "predictions", "info_kb",  "JSON NULL")
        _ensure_column(conn, "predictions", "info_llm", "JSON NULL")
        _ensure_column(conn, "predictions", "chat",     "JSON NULL")

@app.on_event("startup")
def on_startup():
    try:
        ensure_schema()
    except Exception as e:
        print("⚠️ 啟動時修補資料表失敗（略過）：", e)

# ========= Auth =========
class RegisterIn(BaseModel):
    username: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(pw):
    return pwd_context.hash(pw)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="無效的認證資訊",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise cred_exc
    except JWTError:
        raise cred_exc
    user = get_user_by_username(db, username)
    if user is None:
        raise cred_exc
    return user

def _safe_ext(filename: str, content_type: Optional[str]) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return ext
    if content_type == "image/png":
        return ".png"
    return ".jpg"

# ========= YOLO =========
model = YOLO(r"C:\Users\user\runs\classify\train\weights\best.pt")
model.fuse()

# 這裡寫什麼只是顯示名稱，不影響模型輸出順序
class_names = [
    "Clam", "Crab", "Dolphin", "Eel", "Fish", "Jelly Fish",
    "Lobster", "Octopus", "Otter", "Puffer", "Sea Horse",
    "Sea Ray", "Sea Turtle", "Sea Urchin", "Seal", "Shark", "Shrimp",
    "Squid", "Starfish", "Whale", "Other"
]
TOPK = 5
OTHER_CLASS_NAME = "Other"
OTHER_MIN_CONF = 0.7   # 低於這個信心值就判為 Other（可自行調整）

# ========= Ollama / Animal DB 設定 =========
USE_OLLAMA = True
OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.1:8b"

ANIMAL_TABLE = "animal"  # ocean 資料庫裡的 animal 表

ANIMAL_NAME_COLS    = ["english_name", "en_name", "name_en", "class_name", "label_en", "label", "name"]
ANIMAL_CHINESE_COLS = ["chinese_name", "zh_name", "name_zh"]
ANIMAL_SCI_COLS     = ["scientific_name", "latin_name", "binomial", "binomial_name", "sci_name", "scientific", "scient_name",
    "species", "species_name", "taxonomy_scientific", "scientific_names"]
ANIMAL_INFO_COLS = [
    ("分類",           ["family", "taxonomy", "class", "order"]),
    ("常見棲地與分布", ["habitat"]),
    ("生物型態",       ["morphology", "morphology_notes", "body_form"]),
    ("行為",           ["behavior", "behaviour", "behavior_notes", "activity", "habit"]),
    ("是否可食用",     ["edible", "edibility"]),
    ("是否保育類",     ["protected", "conservation_status"]),
]

# ---- 名稱標準化：Fish / Tropical fish 都視為同一群 ----
def _key(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "").replace("-", "")

# 標準顯示 / DB 搜尋用名稱；force=True 代表強制覆蓋顯示名稱
LABEL_CANON = {
    "Fish":           {"en": "Tropical Fish", "zh": "熱帶魚"},
    "Tropical fish":  {"en": "Tropical Fish", "zh": "熱帶魚"},
    "Tropical Fish":  {"en": "Tropical Fish", "zh": "熱帶魚"},
    "Puffer":         {"en": "Pufferfish",    "zh": "河豚", "force": True},
    "Pufferfish":     {"en": "Pufferfish",    "zh": "河豚", "force": True},
    "Whale":          {"en": "Killer Whale",  "zh": "虎鯨", "force": True},
    "Killer Whale":   {"en": "Killer Whale",  "zh": "虎鯨", "force": True},
    "Orca":           {"en": "Killer Whale",  "zh": "虎鯨", "force": True},
}

LABEL_SYNONYMS = {
    "fish": {
        "keys": {_key("fish"), _key("tropical fish"), _key("tropical-fish")},
        "canon": "Fish",
    },
    "puffer": {
        "keys": {
            _key("puffer"), _key("pufferfish"), _key("puffer-fish"), _key("puffer fish"),
            _key("blowfish"), _key("balloonfish"), _key("porcupinefish"), _key("fugu"),
        },
        "canon": "Puffer",
    },
}

_OTHER_KEYS = {_key("other"), _key("others"), _key("unknown"), _key("unidentified"), _key("misc")}
def _is_other(label: str) -> bool:
    return _key(label) in _OTHER_KEYS

def _canon_for(label: str) -> Optional[Dict[str, str]]:
    k = _key(label)
    for group in LABEL_SYNONYMS.values():
        if k in group["keys"]:
            return LABEL_CANON.get(group["canon"])
    return LABEL_CANON.get(label) or LABEL_CANON.get(label.title())



def _norm_label(lbl: str) -> List[str]:
    base = (lbl or "").strip()
    out = {base, base.lower(), base.replace(" ", ""), base.replace(" ", "-")}
    # Fish 群
    if _key(base) in {"fish", "tropicalfish"}:
        out.update({"Fish", "fish", "Tropical Fish", "Tropical fish", "tropicalfish"})
    # Puffer 群
    if _key(base) in {"puffer", "pufferfish", "blowfish", "fugu"}:
        out.update({"Puffer", "puffer", "Pufferfish", "pufferfish", "Puffer fish", "puffer-fish", "Blowfish", "Fugu"})
    # Whale / Orca 群
    if _key(base) in {"whale", "killerwhale", "killer-whale", "orcinusorca", "orca"}:
        out.update({"Whale", "whale", "Killer Whale", "killer whale", "killer-whale", "Orca", "orca", "Orcinus orca"})

    # 其他常見同義
    mapping = {
        "Jelly Fish": ["Jelly Fish", "Jellyfish"],
        "Sea Horse":  ["Sea Horse", "Seahorse"],
        "Sea Ray":    ["Sea Ray", "Ray", "Stingray"],
        "Sea Turtle": ["Sea Turtle", "Sea turtle", "Turtle (sea)"],
        "Sea Urchin": ["Sea Urchin", "Sea-Urchin", "Urchin"],
    }
    if base in mapping:
        out.update(mapping[base])
    return list(out)


# ---- 自動糾正：若把「可食用」與「保育類」寫反，幫忙交換 ----
_EDIBLE_HINTS = ["食用", "食材", "料理", "捕撈", "捕捞", "養殖", "养殖", "可食", "烹調", "烹调", "市場", "市場價值", "海鮮", "海产"]
_PROTECTED_HINTS = ["保育", "保護", "保護類", "保育類", "受威脅", "瀕危", "瀕臨", "紅皮書", "IUCN", "CITES", "法規", "禁止", "名錄", "名录"]

def _looks(text: str, keys: list[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return any(k in text for k in keys)

def _fix_edible_protected(out: dict) -> dict:
    e = out.get("是否可食用")
    p = out.get("是否保育類")
    if _looks(p, _EDIBLE_HINTS) and not _looks(p, _PROTECTED_HINTS) and not _looks(e, _EDIBLE_HINTS):
        out["是否可食用"], out["是否保育類"] = p, (e if _looks(e, _PROTECTED_HINTS) else "未知")
    e = out.get("是否可食用"); p = out.get("是否保育類")
    if _looks(e, _PROTECTED_HINTS) and not _looks(e, _EDIBLE_HINTS) and not _looks(p, _PROTECTED_HINTS):
        out["是否保育類"], out["是否可食用"] = e, (p if _looks(p, _EDIBLE_HINTS) else "未知")
    return out

# ✅ 要求 LLM 一定要輸出的欄位
REQUIRED_KEYS = ["中文名稱", "英文名稱", "分類", "常見棲地與分布", "生物型態", "行為"]

def _ensure_keys(d: dict, kb: dict) -> dict:
    if not isinstance(d, dict):
        d = {}
    out = dict(d)
    for k in REQUIRED_KEYS:
        v = out.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            out[k] = (kb or {}).get(k) or "未知"
    return out

def fetch_animal_info(db: Session, label: str) -> Dict[str, Any]:
    """從 ocean.animal 撈資料；查不到也會回傳標準名，避免顯示未知。"""

    # 模型結果為 Other/Unknown → 不查 DB，直接回基本資訊
    if _is_other(label):
        return {
            "中文名稱": "其他",
            "英文名稱": "Other",
            "分類": "Other",
            "常見棲地與分布": "未知",
            "生物型態": "未知",
            "行為": "未知",
            "是否可食用": "未知",
            "是否保育類": "未知",
        }

    canon = _canon_for(label)  # 例如 Fish → Tropical Fish、Puffer → Pufferfish/河豚

    # -------- 讀欄位清單，建立「寬鬆比對」查找表 --------
    def _norm_col(s: str) -> str:
        return (s or "").lower().replace(" ", "").replace("_", "").replace("-", "")

    conn = db.connection()
    cols_rows = conn.execute(text("""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t
    """), {"t": ANIMAL_TABLE}).all()
    cols = [r[0] for r in cols_rows]

    # 允許以「原字、小寫、去底線/空白/連字號」任一鍵查到真實欄位名
    col_lut: Dict[str, str] = {}
    for c in cols:
        col_lut[c] = c
        col_lut[c.lower()] = c
        col_lut[_norm_col(c)] = c

    def _pick(row: dict, tokens: list[str]) -> Optional[str]:
        """大小寫/底線/空白/連字號/複數等模糊找值；回傳去頭尾空白字串或 None。"""
        if not row:
            return None
        tried = set()
        for t in tokens:
            variants = {
                t, t.lower(), t.upper(), t.title(),
                t.replace(" ", ""), t.replace("_", ""), t.replace("-", ""),
                t + "s", (t + "s").lower(), t + "es", (t + "es").lower(),
            }
            for v in variants:
                key = _norm_col(v)
                if key in tried:
                    continue
                tried.add(key)
                real = col_lut.get(key) or col_lut.get(v.lower())
                if real and real in row and row[real] is not None:
                    sval = str(row[real]).strip()
                    if sval:
                        return sval
        return None

    if not cols:
        out = {
            "英文名稱": (canon or {}).get("en") or label,
            "中文名稱": (canon or {}).get("zh"),
            "分類": label,
            "常見棲地與分布": "未知",
            "生物型態": "未知",
            "行為": "未知",
            "是否可食用": "未知",
            "是否保育類": "未知",
        }
        if canon and canon.get("force"):
            out["英文名稱"] = canon.get("en", out["英文名稱"])
            if canon.get("zh"):
                out["中文名稱"] = canon["zh"]
        return out

    # -------- 先找到資料列（用多個可能的英文名欄位去比對）--------
    def _query_by_names(names: List[str]):
        if not names:
            return None
        name_keys = ["english_name", "en_name", "name_en", "class_name", "label_en", "label", "name"]
        name_cols = [col_lut.get(_norm_col(k)) for k in name_keys if col_lut.get(_norm_col(k))]
        if not name_cols:
            return None

        where_sql, params = [], {}
        for i, nm in enumerate(names):
            nm_trim = nm.strip()
            nm_nospace = nm_trim.replace(" ", "")
            for nc in name_cols:
                # 精確小寫比對 + 去空白比對
                where_sql.append(f"LOWER(TRIM({nc})) = :n{i}_{nc}")
                params[f"n{i}_{nc}"] = nm_trim.lower()
                where_sql.append(f"LOWER(REPLACE({nc}, ' ', '')) = :nx{i}_{nc}")
                params[f"nx{i}_{nc}"] = nm_nospace.lower()

        sql = f"SELECT * FROM {ANIMAL_TABLE} WHERE " + " OR ".join(f"({w})" for w in where_sql) + " LIMIT 1"
        return conn.execute(text(sql), params).mappings().first()

    row = _query_by_names(_norm_label(label))
    if not row and canon:
        row = _query_by_names(_norm_label(canon["en"]))

    if not row:
        return {
            "英文名稱": (canon or {}).get("en") or label,
            "中文名稱": (canon or {}).get("zh"),
            "分類": label,
            "常見棲地與分布": "未知",
            "生物型態": "未知",
            "行為": "未知",
            "是否可食用": "未知",
            "是否保育類": "未知",
        }

    row = dict(row)

    # -------- 名稱與固定欄位 --------
    en = _pick(row, ["english_name", "en_name", "name_en", "class_name", "label_en", "label", "name"]) \
         or (canon or {}).get("en") or label
    zh = _pick(row, ["chinese_name", "zh_name", "name_zh", "chinese", "zh"]) or (canon or {}).get("zh")

    out = {
        "英文名稱": en,
        "中文名稱": zh,
        "分類": label,
    }

    # ✅ 常見棲地與分布：只讀取 habitat（容忍 Habitats 等變體）
    out["常見棲地與分布"] = _pick(row, ["habitat"]) or "未知"

    # ✅ 生物型態：morphology（單行）
    out["生物型態"] = _pick(row, ["morphology"]) or "未知"

    # ✅ 行為：behavior（單行）
    out["行為"] = _pick(row, ["behavior"]) or "未知"

    # 其他欄位
    out["是否可食用"] = _pick(row, ["edible", "edibility"]) or "未知"
    out["是否保育類"] = _pick(row, ["protected", "conservation_status"]) or "視種類 / 依地區法規而異"

    # 套用強制覆蓋（如 Puffer → Pufferfish / 河豚）
    if canon and canon.get("force"):
        out["英文名稱"] = canon.get("en", out["英文名稱"])
        if canon.get("zh"):
            out["中文名稱"] = canon["zh"]

    return out




    def first_exist(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols: return c
        return None

    col_en = first_exist(ANIMAL_NAME_COLS)
    col_zh = first_exist(ANIMAL_CHINESE_COLS)
    col_sci = first_exist(ANIMAL_SCI_COLS)

    def _query_by_names(names: List[str]):
        if not col_en or not names:
            return None
        where_sql, params = [], {}
        for i, name in enumerate(names):
            where_sql.append(f"LOWER(TRIM({col_en})) = :en{i}")
            params[f"en{i}"] = name.strip().lower()
            where_sql.append(f"LOWER(REPLACE({col_en}, ' ', '')) = :enx{i}")
            params[f"enx{i}"] = name.replace(" ", "").lower()
        sql = f"SELECT * FROM {ANIMAL_TABLE} WHERE " + " OR ".join(f"({w})" for w in where_sql) + " LIMIT 1"
        return conn.execute(text(sql), params).mappings().first()

    # 先用原標籤 + 同義詞
    row = _query_by_names(_norm_label(label))
    # 再用標準英文名找（Fish → Tropical Fish、Puffer → Pufferfish）
    if not row and canon:
        row = _query_by_names(_norm_label(canon["en"]))

    if not row:
        out = {
            "英文名稱": (canon or {}).get("en") or label,
            "中文名稱": (canon or {}).get("zh"),
            "分類": label,
        }
        if canon and canon.get("force"):
            out["英文名稱"] = canon.get("en", out.get("英文名稱"))
            if canon.get("zh"): out["中文名稱"] = canon["zh"]
        return out

    row = dict(row)
    out: Dict[str, Any] = {}
    out["英文名稱"] = (row.get(col_en) if col_en else None) or (canon or {}).get("en") or label
    out["中文名稱"] = (row.get(col_zh) if col_zh else None) or (canon or {}).get("zh")
    if col_sci:
        out["學名"] = row.get(col_sci)

    for target_key, cands in ANIMAL_INFO_COLS:
        for c in cands:
            if c in row and row[c]:
                out[target_key] = row[c]
                break

    out.setdefault("分類", label)

    # ★ 若 canonical 標記了 force，覆蓋顯示名稱（解決 DB 寫成「膠鰓魚」的情況）
    if canon and canon.get("force"):
        out["英文名稱"] = canon.get("en", out.get("英文名稱"))
        if canon.get("zh"): out["中文名稱"] = canon["zh"]

    return out


def _compose_prompt_from_db(label: str, kb: Dict[str, Any], topk: List[Dict[str, Any]]) -> str:
    def fmt_topk(arr: List[Dict[str, Any]]) -> str:
        items = []
        for d in (arr or [])[:5]:
            try:
                items.append(f"{d.get('class_name')}({float(d.get('confidence',0)):.2f})")
            except Exception:
                items.append(f"{d.get('class_name')}(?)")
        return ", ".join(items)
    return (
        "你是海洋生物學助理。以下是資料庫提供的 JSON（具最高優先權），"
        "請**只根據這份 JSON** 回答，並用**繁體中文**輸出同樣的 JSON 欄位："
        '["中文名稱","英文名稱","分類","常見棲地與分布","生物型態","是否可食用","是否保育類"]。'
        "若某欄位在資料庫沒有，請填「未知」或「視種類 / 依地區法規而異」。\n\n"
        f"辨識主結果：{label}\nTop-5：{fmt_topk(topk)}\n\n"
        f"【資料庫 JSON】：\n{json.dumps(kb, ensure_ascii=False)}"
    )

def ask_ollama_with_context(kb: Dict[str, Any], label: str, topk: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not USE_OLLAMA:
        return _ensure_keys({}, kb)
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are a concise marine biology assistant. Reply ONLY JSON in Traditional Chinese."},
                {"role": "user", "content": _compose_prompt_from_db(label, kb, topk)},
            ],
            "options": {"temperature": 0.1},
            "format": "json",
            "stream": False,
        }
        r = http.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        raw = r.json().get("message", {}).get("content", "{}")
        data = json.loads(raw)
        return _ensure_keys(_fix_edible_protected(data), kb)
    except Exception as e:
        return _ensure_keys({"補充說明": f"Ollama 解析失敗：{e}"}, kb)

# ========= 健康檢查 =========
@app.get("/ollama/health")
def ollama_health():
    try:
        r = http.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        ok = r.ok
    except Exception as e:
        return {"ok": False, "error": str(e), "model": OLLAMA_MODEL}
    return {"ok": True, "model": OLLAMA_MODEL}

# ========= 註冊 / 登入 =========
@app.post("/register", status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if get_user_by_username(db, body.username):
        raise HTTPException(400, "使用者已存在")
    user = User(username=body.username, hashed_password=get_password_hash(body.password))
    db.add(user)
    db.commit()
    return {"message": "註冊成功"}

@app.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_username(db, form.username)
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(400, "帳號或密碼錯誤")
    token = create_access_token({"sub": user.username})
    return TokenOut(access_token=token)

# ========= 推論 + 寫歷史（含 info_kb / info_llm） =========
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(True, description="是否呼叫 Ollama 產生解說"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "請上傳圖片檔案"}, status_code=400)

    # 讀檔
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return JSONResponse({"error": "無法辨識圖片檔案，可能檔案已損壞"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"讀檔錯誤: {str(e)}"}, status_code=400)

    # 計算下一個 seq（防併發唯一索引策略：若衝突就重試）
    attempt = 0
    while True:
        attempt += 1
        next_seq = db.query(func.coalesce(func.max(Prediction.seq), 0))\
                     .filter(Prediction.user_id == current_user.id)\
                     .scalar() + 1
        seq_str = f"{next_seq:04d}"

        # 存檔（使用者/日期分目錄；檔名加 seq 前綴）
        day = datetime.now().strftime("%Y%m%d")
        user_rel_dir = Path(current_user.username) / day
        user_abs_dir = BASE_UPLOAD_DIR / user_rel_dir
        user_abs_dir.mkdir(parents=True, exist_ok=True)

        ext = _safe_ext(file.filename or "", file.content_type or "")
        filename = f"{seq_str}_{datetime.now().strftime('%H%M%S')}_{uuid4().hex[:8]}{ext}"
        abs_path = user_abs_dir / filename
        with open(abs_path, "wb") as f:
            f.write(img_bytes)

        rel_path = str((user_rel_dir / filename).as_posix())
        file_url = f"/files/{rel_path}"

        # YOLO 推論
        results = model(img)
        r = results[0]
        probs = r.probs.data if hasattr(r.probs, "data") else r.probs
        probs = probs.detach().cpu().tolist()

        detections: List[Dict[str, Any]] = []
        for i, p in enumerate(probs):
            name = class_names[i] if i < len(class_names) else str(i)
            detections.append({"class_id": i, "class_name": name, "confidence": float(p)})
        # YOLO 推論後…
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        top_result = detections[0]
        topk = detections[:TOPK]
        
        # ★ 置信度 ≤ 門檻 → 強制標為 Other
        if top_result["confidence"] <= OTHER_MIN_CONF:
            other_idx = class_names.index(OTHER_CLASS_NAME)
            top_result = {
                "class_id": other_idx,
                "class_name": OTHER_CLASS_NAME,
                "confidence": float(top_result["confidence"])
            }
            topk = [top_result] + [d for d in detections if d["class_name"] != OTHER_CLASS_NAME][:TOPK-1]
        
        # DB 知識庫
        info_kb = fetch_animal_info(db, top_result["class_name"]) or {}
        
        # ★ 套用 canonical（含 force 覆蓋）
        canon = _canon_for(top_result["class_name"])
        if canon:
            if canon.get("force"):
                info_kb["英文名稱"] = canon["en"]
                if canon.get("zh"): info_kb["中文名稱"] = canon["zh"]
            else:
                info_kb.setdefault("英文名稱", canon["en"])
                if canon.get("zh"): info_kb.setdefault("中文名稱", canon["zh"])
        info_kb.setdefault("分類", top_result["class_name"])
        _fix_edible_protected(info_kb)
        
        # Other 不呼叫 LLM
        if explain and not _is_other(top_result["class_name"]):
            info_llm = ask_ollama_with_context(info_kb, top_result["class_name"], topk)
        else:
            info_llm = None


        # 寫 DB
        rec = Prediction(
            user_id=current_user.id,
            seq=next_seq,
            image_rel_path=rel_path,
            image_url=file_url,
            top_class=top_result["class_name"],
            top_conf=f'{top_result["confidence"]:.6f}',
            topk=topk,
            all_predictions=detections,
            info_kb=info_kb or None,
            info_llm=info_llm or None,
            chat=[],
        )
        db.add(rec)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            if attempt < 3:
                continue
            raise
        except OperationalError as e:
            db.rollback()
            msg = str(e).lower()
            if "1054" in msg or "unknown column" in msg:
                ensure_schema()
                db.add(rec)
                db.commit()
            else:
                raise
        break

    return {
        "seq": next_seq,
        "top_prediction": top_result,
        "topk_predictions": topk,
        "all_predictions": detections,
        "saved_image_path": str(abs_path),
        "image_url": file_url,
        "username": current_user.username,
        "record_id": rec.id,
        "kb_info": info_kb,
        "llm_info": info_llm,
        "chat": rec.chat or [],
    }

# ========= 歷史 =========
@app.get("/history")
def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.seq.desc())
        .offset(offset).limit(limit)
    )
    rows = q.all()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "seq": r.seq,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "image_url": r.image_url,
            "image_rel_path": r.image_rel_path,
            "top_class": r.top_class,
            "top_conf": r.top_conf,
            "topk": r.topk,
            "info_kb": r.info_kb,
            "info_llm": r.info_llm,
            "chat": r.chat or [],
        })
    return {"items": data, "limit": limit, "offset": offset}

# 針對單筆補產生解說（若早期資料沒有）
@app.post("/history/{pred_id}/explain")
def explain_history(
    pred_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rec = db.query(Prediction).filter(
        Prediction.id == pred_id,
        Prediction.user_id == current_user.id
    ).first()
    if not rec:
        raise HTTPException(404, "找不到這筆歷史")
    info_kb  = rec.info_kb or fetch_animal_info(db, rec.top_class) or {}
    info_llm = ask_ollama_with_context(info_kb, rec.top_class, rec.topk or [])
    rec.info_kb = info_kb or None
    rec.info_llm = info_llm or None
    db.commit()
    return {"id": rec.id, "seq": rec.seq, "info_kb": rec.info_kb, "info_llm": rec.info_llm}

# ========= 聊天 =========
class ChatIn(BaseModel):
    record_id: int
    question: str
    free: Optional[bool] = None  # True=自由聊；False/None=鎖定物種

class ChatQ(BaseModel):
    question: str
    free: bool = False

def _fmt_topk_inline(topk: List[Dict[str, Any]]) -> str:
    parts = []
    for d in (topk or [])[:5]:
        try:
            parts.append(f"{d.get('class_name')}({float(d.get('confidence',0)):.2f})")
        except Exception:
            parts.append(f"{d.get('class_name')}(?)")
    return ", ".join(parts)

def _subject_name(rec: Prediction, kb: Dict[str, Any]) -> str:
    zh = (kb or {}).get("中文名稱")
    en = (kb or {}).get("英文名稱")
    sci = (kb or {}).get("學名")
    base = zh or rec.top_class
    extras = [x for x in [en, sci] if x]
    return base if not extras else f"{base}（{'，'.join(extras)}）"

def _chat_core(rec: Prediction, question: str, db: Session, free: Optional[bool] = None) -> str:
    info_kb = rec.info_kb or fetch_animal_info(db, rec.top_class) or {}
    rec.info_kb = info_kb or None

    subject = _subject_name(rec, info_kb)
    topk_inline = _fmt_topk_inline(rec.topk or [])

    system_msg = (
        "你是專業且嚴謹的海洋生物學助理，使用繁體中文，回答要具體、正確、可操作，避免胡亂猜測。"
        "如涉及毒性、食用安全或法規，請清楚標註風險並提醒「依地區法規而異」。"
    )

    if free:
        context_msg = (
            f"主題物種：{subject}。\n"
            f"Top-5：{topk_inline}\n\n"
            "你可以自由回答使用者問題；以下的資料庫 JSON 只作為參考，不必受限於它。\n"
            f"資料庫 JSON（參考）：{json.dumps(info_kb, ensure_ascii=False)}"
        )
    else:
        context_msg = (
            f"主題物種：{subject}。之後的問題一律視為針對此物種。\n"
            f"Top-5：{topk_inline}\n\n"
            "請以該物種為主題作答，可結合可靠的一般海洋生物學知識與經驗，不限於資料庫內容；"
            "下列資料庫 JSON 主要用來校對名稱/分類等事實，若有衝突以資料庫為優先。"
            "若真的無資料或仍有爭議，才回覆「不確定」或「依地區法規而異」。\n"
            f"資料庫 JSON（校對用）：{json.dumps(info_kb, ensure_ascii=False)}"
        )

    history = rec.chat or []
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": context_msg}]
    for m in history[-8:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": f"（聚焦於 {subject}）{question}"})

    r = http.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "options": {"temperature": 0.2}, "stream": False},
        timeout=120
    )
    r.raise_for_status()
    answer = r.json().get("message", {}).get("content", "").strip() or "（沒有產生內容）"

    now = datetime.utcnow().isoformat()
    history.append({"role": "user", "content": question, "ts": now})
    history.append({"role": "assistant", "content": answer, "ts": now})
    rec.chat = history
    db.commit()
    return answer

# /chat（body 內帶 record_id）
@app.post("/chat")
def chat_with_ollama(body: ChatIn = Body(...),
                     db: Session = Depends(get_db),
                     current_user: User = Depends(get_current_user)):
    rec = db.query(Prediction).filter(
        Prediction.id == body.record_id,
        Prediction.user_id == current_user.id
    ).first()
    if not rec:
        raise HTTPException(404, "找不到這筆歷史")
    try:
        ans = _chat_core(rec, body.question, db, free=body.free)
    except Exception as e:
        raise HTTPException(502, f"Ollama 錯誤：{e}")
    return {"record_id": rec.id, "answer": ans, "messages": rec.chat[-10:]}

# RESTful：/history/{id}/chat
@app.post("/history/{rid}/chat")
def chat_with_ollama_for_record(rid: int,
                                body: ChatQ,
                                db: Session = Depends(get_db),
                                current_user: User = Depends(get_current_user)):
    rec = db.query(Prediction).filter(
        Prediction.id == rid,
        Prediction.user_id == current_user.id
    ).first()
    if not rec:
        raise HTTPException(404, "找不到這筆歷史")
    try:
        ans = _chat_core(rec, body.question, db, free=body.free)
    except Exception as e:
        raise HTTPException(502, f"Ollama 錯誤：{e}")
    return {"record_id": rec.id, "answer": ans, "messages": rec.chat[-10:]}

# =========（可選）根路由 =========
@app.get("/")
def root():
    return {"ok": True, "service": "Ocean AI API", "has_ollama": USE_OLLAMA, "model": OLLAMA_MODEL}
