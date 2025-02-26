import os
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from app_docs import get_documents  # ดึงข้อมูลจาก app_docs.py

# 1. โหลด environment variables จากไฟล์ .env
load_dotenv()

# 2. เริ่มต้น Qdrant (แบบ In-Memory)
qdrant_client = QdrantClient(":memory:")  # ใช้ In-Memory

# สร้าง Collection สำหรับเก็บเวกเตอร์
qdrant_client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # ใช้ 384-D embedding
)

# 3. ดึงข้อมูลเอกสารจาก app_docs.py
def prepare_documents():
    documents = get_documents()
    return [doc.strip() for doc in documents if doc.strip()]

# 4. แปลงข้อความเป็นเวกเตอร์ และเพิ่มลงใน Qdrant
def add_documents_to_qdrant(documents):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # โหลดโมเดลสำหรับทำ Embedding
    vectors = embedding_model.encode(documents).tolist()  # แปลงข้อความเป็นเวกเตอร์

    # เพิ่มข้อมูลลง Qdrant
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": documents[i]}) for i in range(len(documents))]
    qdrant_client.upsert(collection_name="documents", points=points)

# 5. สร้างฟังก์ชันการค้นหาเอกสาร
def search_documents(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedding_model.encode([query])[0].tolist()
    search_results = qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=2  # ดึงเอกสารที่เกี่ยวข้อง 2 อันดับแรก
    )
    return [hit.payload["text"] for hit in search_results]

# 6. สร้างฟังก์ชันการสร้างคำตอบด้วย Groq
def generate_answer(query):
    # ค้นหาข้อมูลที่เกี่ยวข้องจาก Qdrant
    retrieved_docs = search_documents(query)

    # รวมข้อมูลเข้าไปใน Prompt
    context = "\n".join(retrieved_docs)
    prompt = [
        {"role": "system", "content": "คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับข้อมูลในเอกสาร จงตอบคำถามอย่างกระชับและถูกต้อง"},
        {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}\n\nคำตอบ:"}
    ]

    # เรียก Groq API
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=prompt
    )

    return response.choices[0].message.content

# 7. สร้างอินเทอร์เฟซด้วย Streamlit
def main():
    st.title("RAG Chatbot เกี่ยวกับคาเฟ่ในอำเภอเมืองจังหวัดน่าน")
    st.write("สวัสดี Chatbot ที่ช่วยตอบคำถามจากเอกสารที่มีอยู่")

    # ดึงข้อมูลเอกสาร
    documents = prepare_documents()

    # เพิ่มข้อมูลลง Qdrant
    add_documents_to_qdrant(documents)
    st.success("เอกสารถูกประมวลผลและพร้อมใช้งานแล้ว!")

    # รับคำถามจากผู้ใช้
    query = st.text_input("คุณ: ", placeholder="พิมพ์คำถามของคุณที่นี่...")

    if st.button("ส่ง"):
        if query:
            # สร้างคำตอบ
            answer = generate_answer(query)
            st.write("Bot:", answer)
        else:
            st.warning("กรุณาพิมพ์คำถามก่อนส่ง")

# 8. เรียกใช้แอปพลิเคชัน
if __name__ == "__main__":
    main()
