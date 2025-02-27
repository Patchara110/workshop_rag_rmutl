import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import PyPDF2  # สำหรับอ่าน PDF
from groq import Groq

# 1. ฟังก์ชันอ่าน PDF และแปลงเป็นข้อความ
def extract_text_from_pdf(pdf_path):
    text = ""
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    else:
        st.error(f"ไม่พบไฟล์ PDF ที่ {pdf_path}")
    return text.strip()

# 2. โหลดข้อมูลจาก PDF
pdf_path = "pdf/คาเฟ่ในอำเภอเมืองจังหวัดน่าน.pdf"
documents = [extract_text_from_pdf(pdf_path)]  # ต้องเป็น List เพื่อให้ encode ได้

# 3. เชื่อมต่อ Qdrant
qdrant_client = QdrantClient(":memory:")  # ใช้ memory (เปลี่ยนเป็นเซิร์ฟเวอร์จริงถ้ามี)

# 4. แปลงข้อความเป็นเวกเตอร์ และเพิ่มลงใน Qdrant
def add_documents_to_qdrant(documents):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedding_model.encode(documents).tolist()
    
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": documents[i]}) for i in range(len(documents))]
    qdrant_client.upsert(collection_name="documents", points=points)

# 5. สร้างฟังก์ชันการค้นหาเอกสาร
def search_documents(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedding_model.encode([query])[0].tolist()
    
    search_results = qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=2
    )
    
    return [hit.payload["text"] for hit in search_results if "text" in hit.payload]

# 6. สร้างฟังก์ชันการสร้างคำตอบด้วย Groq
def generate_answer(query):
    retrieved_docs = search_documents(query)

    if not retrieved_docs:
        return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้อง"

    context = "\n".join(retrieved_docs)
    prompt = [
        {"role": "system", "content": "คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับคาเฟ่ในอำเภอเมือง จังหวัดน่าน ตอบคำถามให้กระชับและถูกต้อง"},
        {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}\n\nคำตอบ:"}
    ]

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=prompt
    )

    return response.choices[0].message.content

# 7. สร้างอินเทอร์เฟซด้วย Streamlit
def main():
    st.title("RAG Chatbot เกี่ยวกับคาเฟ่ในตัวอำเภอเมือง จังหวัดน่าน")
    st.write("สวัสดี! Chatbot นี้ช่วยให้ข้อมูลเกี่ยวกับคาเฟ่ในตัวอำเภอเมือง จังหวัดน่าน")

    # เพิ่มข้อมูลเอกสารลงใน Qdrant (ทำครั้งเดียว)
    if st.button("โหลดข้อมูลจาก PDF"):
        add_documents_to_qdrant(documents)
        st.success("ข้อมูลจาก PDF ถูกโหลดเข้าสู่ระบบเรียบร้อยแล้ว!")

    # รับคำถามจากผู้ใช้
    query = st.text_input("คุณ:", placeholder="พิมพ์คำถามเกี่ยวกับคาเฟ่ที่นี่...")

    if st.button("ส่ง"):
        if query:
            answer = generate_answer(query)
            st.write("Bot:", answer)
        else:
            st.warning("กรุณาพิมพ์คำถามก่อนส่ง")

# 8. เรียกใช้แอปพลิเคชัน
if __name__ == "__main__":
    main()
