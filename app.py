import os
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pypdf import PdfReader

# โหลด environment variables
load_dotenv()

# เริ่มต้น Qdrant (In-Memory)
qdrant_client = QdrantClient(":memory:")

# สร้าง Collection ถ้ายังไม่มี
qdrant_client.recreate_collection(
    collection_name="cafe_documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# ฟังก์ชันอ่าน PDF และแปลงเป็นข้อความ
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# เตรียมข้อมูลจาก PDF
def prepare_documents_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    documents = text.split("\n")  # แยกข้อความตามบรรทัด
    return [doc.strip() for doc in documents if doc.strip()]

# แปลงข้อความเป็นเวกเตอร์ และเพิ่มลง Qdrant
def add_documents_to_qdrant(documents):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedding_model.encode(documents).tolist()
    
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": documents[i]}) for i in range(len(documents))]
    qdrant_client.upsert(collection_name="cafe_documents", points=points)

# ค้นหาเอกสารที่เกี่ยวข้อง
def search_documents(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedding_model.encode([query])[0].tolist()
    
    search_results = qdrant_client.search(
        collection_name="cafe_documents",
        query_vector=query_vector,
        limit=2
    )
    
    return [hit.payload["text"] for hit in search_results if "text" in hit.payload]

# สร้างคำตอบโดยใช้ Groq
def generate_answer(query):
    retrieved_docs = search_documents(query)

    if not retrieved_docs:
        return "ขออภัย ฉันไม่มีข้อมูลที่เกี่ยวข้องกับคำถามนี้"

    context = "\n".join(retrieved_docs)
    prompt = [
        {"role": "system", "content": "คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับคาเฟ่ในอำเภอเมือง จังหวัดน่าน ตอบคำถามอย่างกระชับและถูกต้อง"},
        {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}\n\nคำตอบ:"}
    ]

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=prompt
    )

    return response.choices[0].message.content

# อินเทอร์เฟซ Streamlit
def main():
    st.title("RAG Chatbot: คาเฟ่ในอำเภอเมือง จังหวัดน่าน")
    st.write("Chatbot ที่ช่วยตอบคำถามเกี่ยวกับคาเฟ่ในอำเภอเมือง จังหวัดน่าน")

    # ตั้งค่าไฟล์ PDF ที่จะใช้งาน
    pdf_path = "pdf/คาเฟ่ในอำเภอเมืองจังหวัดน่าน.pdf"

    # ตรวจสอบว่าไฟล์ PDF มีอยู่
    if os.path.exists(pdf_path):
        if st.button("โหลดข้อมูลคาเฟ่จาก PDF"):
            documents = prepare_documents_from_pdf(pdf_path)
            add_documents_to_qdrant(documents)
            st.success("เอกสารถูกโหลดเข้า Qdrant แล้ว! คุณสามารถเริ่มถามคำถามได้")

    # รับคำถามจากผู้ใช้
    query = st.text_input("คุณ:", placeholder="พิมพ์คำถามเกี่ยวกับคาเฟ่ที่นี่...")

    if st.button("ส่ง"):
        if query:
            answer = generate_answer(query)
            st.write("Bot:", answer)
        else:
            st.warning("กรุณาพิมพ์คำถามก่อนส่ง")

# เรียกใช้แอปพลิเคชัน
if __name__ == "__main__":
    main()
