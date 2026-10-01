"""
GaziMind — Ogrenci Yonetmelik Asistani (Chat arayuz)
----------------------------------------------------
Mavi-beyaz universite temasi, sohbet balonlu chatbot arayuzu.

Backend'e (POST /task/) su govdeyi gonderir:
    {"question": <soru>, "history": [{"role": "...", "content": "..."}, ...]}
  - question : kullanicinin son mesaji
  - history  : onceki mesajlar (conversational RAG icin — backend kullanabilir)
    (Mevcut backend 'history'yi yok sayar; sen ekleyince otomatik calisir.)

Calistirma:
    1. Backend:   uvicorn main:app --reload
    2. Arayuz:    streamlit run streamlit_app.py
"""

import os
import uuid

import requests
import streamlit as st

# --- Ayarlar ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/task/")
TIMEOUT = 120

ORNEK_SORULAR = [
    "Gecme harf notlari nasil belirlenir?",
    "Butunleme sinavina girme sarti nedir?",
    "Tez savunmasinda basarisiz olan ogrenci ne yapar?",
    "Yabanci dil hazirlik egitimi zorunlu mu?",
]

st.set_page_config(page_title="GaziMind", page_icon="🎓", layout="centered")


# ============================================================
#  MAVI-BEYAZ UNIVERSITE TEMASI
# ============================================================
st.markdown(
    """
    <style>
      :root {
        --gm-blue: #0b3d91;
        --gm-blue-2: #1e5fd6;
        --gm-blue-soft: #eaf1fb;
      }
      .stApp { background: #f7f9fc; }

      /* Ust banner */
      .gm-hero {
        background: linear-gradient(135deg, #0b3d91 0%, #1e5fd6 100%);
        color: #fff; border-radius: 16px; padding: 22px 26px; margin-bottom: 8px;
        box-shadow: 0 8px 26px rgba(11,61,145,.22);
      }
      .gm-hero h1 { margin: 0; font-size: 27px; font-weight: 800; letter-spacing: -.5px; }
      .gm-hero p  { margin: 5px 0 0; opacity: .92; font-size: 14px; }
      .gm-hero .gm-badge {
        display: inline-block; background: rgba(255,255,255,.18);
        border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
        padding: 3px 12px; font-size: 12px; font-weight: 600; margin-bottom: 8px;
      }

      /* Chat balonlari */
      [data-testid="stChatMessage"] {
        background: #ffffff; border: 1px solid #e2e9f4; border-radius: 14px;
        padding: 4px 6px; box-shadow: 0 1px 6px rgba(11,61,145,.05);
      }
      /* Kullanici balonu (biraz mavi) */
      [data-testid="stChatMessage"]:has(.gm-user) {
        background: var(--gm-blue-soft); border-color: #cfe0f7;
      }

      /* Kaynak kutusu */
      .gm-src {
        background: var(--gm-blue-soft); border-left: 4px solid var(--gm-blue);
        border-radius: 8px; padding: 8px 12px; margin-top: 8px; font-size: 13px;
      }
      .gm-src b { color: var(--gm-blue); }

      /* Butonlar */
      .stButton > button {
        border-radius: 9px; border: 1px solid #cfe0f7; font-size: 13px;
        text-align: left;
      }
      .stButton > button:hover { border-color: var(--gm-blue-2); background: var(--gm-blue-soft); }

      /* Chat input */
      [data-testid="stChatInput"] textarea { border-radius: 10px; }

      /* Sidebar */
      section[data-testid="stSidebar"] { background: var(--gm-blue-soft); }
      section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--gm-blue); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
#  BACKEND CAGRISI
# ============================================================
def cevap_al(soru: str, gecmis: list) -> dict:
    """API'ye soru + gecmis gonderir, {answer, sources} dondurur."""
    govde = {"question": soru, "history": gecmis, "thread_id": st.session_state.thread_id}
    yanit = requests.post(API_URL, json=govde, timeout=TIMEOUT)
    yanit.raise_for_status()
    return yanit.json()


# ============================================================
#  DURUM (hafiza — ekran icin)
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())   # backend bu id ile sohbeti hatirlar


# ============================================================
#  ARAYUZ
# ============================================================

# Ust banner
st.markdown(
    """
    <div class="gm-hero">
      <span class="gm-badge">🎓 Ogrenci Asistani</span>
      <h1>GaziMind</h1>
      <p>Universite yonetmelikleri hakkinda sohbet et — kaynakli, guvenilir cevaplar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("GaziMind")
    st.caption("Gazi Universitesi yonetmelik asistani")
    st.markdown("---")
    st.subheader("Ornek sorular")
    secilen = None
    for s in ORNEK_SORULAR:
        if st.button(s, key=s, use_container_width=True):
            secilen = s
    st.markdown("---")
    if st.button("🗑️ Sohbeti temizle", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())   # yeni sohbet = yeni hafiza
        st.rerun()
    st.caption(
        "Bu asistan SADECE yuklu yonetmeliklere gore cevap verir. "
        "Bilgi yoksa 'bulamadim' der."
    )

# Gecmis mesajlari goster
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🎓" if m["role"] == "assistant" else "🧑‍🎓"):
        # kullanici balonuna renk vermek icin gizli isaret
        if m["role"] == "user":
            st.markdown('<span class="gm-user"></span>', unsafe_allow_html=True)
        st.markdown(m["content"])
        if m.get("sources"):
            liste = "".join(f"• {k}<br>" for k in m["sources"])
            st.markdown(f'<div class="gm-src"><b>Kaynaklar</b><br>{liste}</div>',
                        unsafe_allow_html=True)

# Girdi: chat kutusu YA DA ornek soru butonu
soru = st.chat_input("Sorunuzu yazin...") or secilen

if soru:
    # 1) Kullanici mesajini goster + kaydet
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown('<span class="gm-user"></span>', unsafe_allow_html=True)
        st.markdown(soru)

    # gecmisi (bu mesajdan ONCEKI hali) yakala -> backend'e gonderilecek
    gecmis = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": soru})

    # 2) Asistan cevabi
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Yonetmelikler taraniyor..."):
            try:
                sonuc = cevap_al(soru, gecmis)
                cevap = sonuc.get("answer", "(bos cevap)")
                kaynaklar = sonuc.get("sources", [])

                st.markdown(cevap)
                if kaynaklar:
                    liste = "".join(f"• {k}<br>" for k in kaynaklar)
                    st.markdown(f'<div class="gm-src"><b>Kaynaklar</b><br>{liste}</div>',
                                unsafe_allow_html=True)

                st.session_state.messages.append(
                    {"role": "assistant", "content": cevap, "sources": kaynaklar}
                )

            except requests.exceptions.ConnectionError:
                st.error("Backend'e ulasilamadi. FastAPI acik mi? (uvicorn main:app --reload)")
            except requests.exceptions.Timeout:
                st.error("Zaman asimi. Model cok yavas cevap verdi, tekrar dene.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API hatasi: {e.response.status_code}")
            except Exception as e:
                st.error(f"Beklenmeyen hata: {e}")
