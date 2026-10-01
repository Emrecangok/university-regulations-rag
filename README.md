<p align="center">
  <img src="images/gazi_logo.png" alt="Gazi Üniversitesi" width="140">
</p>

<h1 align="center">GaziMind</h1>

<p align="center">Gazi Üniversitesi yönetmelikleri hakkında kaynaklı cevaplar veren, LangGraph tabanlı bir RAG asistanı.</p>

![GaziMind arayüzü](images/main_screen.png)

## Kullanılan Teknolojiler

- **Google Gemini:** LLM ve embedding modeli
- **LangGraph:** ajan akışı
- **SQLite:** LangGraph checkpointer ile sohbet hafızası. Her kullanıcının konuşması `thread_id` ile saklanır.
- **LangChain + Chroma:** vektör veritabanı ve retrieval
- **Docling:** PDF'leri parse edip başlık yapısına göre chunk'lama
- **FastAPI:** backend API
- **Streamlit:** chat arayüzü
- **Docker Compose:** API ve arayüzü birlikte ayağa kaldırma

## Akış Diyagramı

<img src="images/diagram.png" alt="GaziMind graf akışı" width="420">

Klasik "soru → ara → cevapla" zinciri yerine, her adımda karar veren bir graf kurdum:

1. **Router:** Gelen mesajı sınıflandırır. "Selam" gibi sohbet mesajları doğrudan **Chat** node'una gider. Yönetmelik soruları doküman aramasına yönlenir.
2. **Retrieve Docs:** Chroma'dan soruya en yakın yönetmelik parçalarını getirir.
3. **Grade Docs:** LLM, getirilen parçaların soruyu gerçekten cevaplayıp cevaplamadığını değerlendirir. Sadece kelime benzerliği yeterli sayılmaz.
   - **Alakalıysa:** **Generate** node'u bu parçalara dayanarak, madde ve kaynak göstererek cevap üretir.
   - **Alakasızsa:** **Rewrite Query** soruyu resmî yönetmelik diline çevirir ("okuldan atılmak" → "ilişik kesilmesi") ve arama tekrarlanır.
   - **Deneme limiti dolarsa:** **Fallback** devreye girer. Bilgi uydurulmaz, öğrenci Öğrenci İşleri'ne yönlendirilir.
4. **Summary:** Sohbet belirli bir uzunluğu geçince eski mesajlar özetlenip silinir. Böylece bağlam korunur, token maliyeti düşük kalır.

Her node'un sistem prompt'u `prompts/` klasöründe ayrı bir dosyada tutulur. Router ve grader, Pydantic şemasıyla structured output döndürür.

## Çalıştırma

```bash
pip install -r requirements.txt
python rag.py index                 # PDF'leri indeksle (.env içinde GEMINI_API_KEY gerekli)
uvicorn main:app --port 8000        # API
streamlit run streamlit_app.py      # Arayüz
```
