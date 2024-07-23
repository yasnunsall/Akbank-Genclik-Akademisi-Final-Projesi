import os
from chromadb.config import DEFAULT_TENANT,  DEFAULT_DATABASE, Settings
from chromadb import Client, PersistentClient
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
  model_name = "paraphrase-multilingual-MiniLM-L12-v2"
)

chroma_client = PersistentClient(
  path = "ChromaDBData",
  settings = Settings(),
  tenant = DEFAULT_TENANT,
  database = DEFAULT_DATABASE
)

chroma_collection = chroma_client.get_or_create_collection(
  "Akbank",
  embedding_function = embedding_function
)

def build_chatBot(system_instruction):
  model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=system_instruction)
  chat = model.start_chat(history=[])
  return chat

def generate_LLM_answer(prompt, context, chat):
  response = chat.send_message( prompt + context)
  return response.text

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

system_prompt = """ Sen, Akbank'ın kendisiymiş gibi konuşan bir dijital asistansın. Amacın, kullanıcıların sorularına resmi bir dille ve dosyada bulunan bilgilere dayanarak cevap vermektir.

1. Sana kullanıcının sorusunu ve ilgili metin alıntılarını sağlayacağım.
2. Görevin, yalnızca sağlanan metin alıntılarını kullanarak Akbank adına cevap vermektir.
3. Yanıtı oluştururken şu kurallara dikkat et:
   - Sağlanan metin alıntısında açıkça yer alan bilgileri kullan.
   - Sağlanan metin alıntısında cevap bulunmuyorsa, şu ifadeyi kullan: 'Ne yazık ki bu soruya cevap verebilecek bilgiye sahip değilim. Sorduğunuz konu hakkında daha fazla ayrıntı sağlayabilir veya farklı bir soru sorabilirsiniz.'
   - Metin alıntısında açıkça bulunmayan cevapları tahmin etmeye veya uydurmaya çalışma.
4. Sohbet oturumu ile ilgili genel sorular (sohbeti özetle, soruları listele gibi) için sağlanan metin alıntısını kullanma. Bu tür sorulara doğrudan cevap ver.
5. Yanıtı, kullanıcının sorduğu dilde oluştur.
6. Yanıtın içinde link varsa tek ve tıklanabilir bir link olmasını sağla.
7. Kullanıcıya her zaman yardımcı olmaya çalış, ancak mevcut bilgilere dayanmayan yanıtlardan kaçın.
8. Yanıtın içinde link bulunuyorsa tıklanabilir hale getir.

Eğer hazırsan, sana kullanıcının sorusunu ve ilgili metin alıntısını sağlıyorum.
"""

def build_chatBot(system_instruction):
  model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=system_instruction)
  chat = model.start_chat(history=[])
  return chat

RAG_LLM = None

def initialize_rag_system():
    global RAG_LLM
    RAG_LLM = build_chatBot(system_prompt)

def retrieveDocs(chroma_collection, query, n_results=10, return_only_docs=False):
    results = chroma_collection.query(query_texts=[query],
                                      include= [ "documents", "metadatas", 'distances' ],
                                      n_results=n_results)

    if return_only_docs:
        return results['documents'][0]
    else:
        return results

def generateAnswer(query):
    retrieved_documents= retrieveDocs(chroma_collection, query, 10, return_only_docs=True)
    prompt = "Soru: "+ query
    context = "\n Alıntı: "+ "\n".join(retrieved_documents)

    output = generate_LLM_answer(prompt, context, RAG_LLM)
    return output