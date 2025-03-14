import os
import re
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# 1. โหลด environment variables จากไฟล์ .env
load_dotenv()

# 2. เริ่มต้น Qdrant (แบบ In-Memory)
qdrant_client = QdrantClient(":memory:")  # ใช้ In-Memory

# สร้าง Collection สำหรับเก็บเวกเตอร์
qdrant_client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # ใช้ 384-D embedding
)

# 3. ฟังก์ชันสำหรับอ่านไฟล์ PDF และดึงข้อความ (แก้ไขปัญหาลิงก์หาย)
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    url_pattern = r'https?://[^\s]+'  # ดึง URL ออกมาให้ครบ

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

            # ดึง URL จากหน้า PDF (แก้ปัญหาลิงก์ขาดหาย)
            urls = re.findall(url_pattern, page_text)
            if urls:
                text += "\n".join(urls) + "\n"

    return text

# 4. เตรียมข้อมูลเอกสารจากไฟล์ PDF (แยกแต่ละร้าน)
def prepare_documents_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    # ปรับปรุงการแยกข้อมูลร้านค้าโดยใช้ regex ที่แม่นยำกว่า
    documents = re.split(r"---[\s]*ชื่อร้าน:\s*", text)
    documents = ["ชื่อร้าน:" + doc.strip() for doc in documents if doc.strip()]
    return documents

# 5. แปลงข้อความเป็นเวกเตอร์ และเพิ่มลงใน Qdrant
def add_documents_to_qdrant(documents):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # โหลดโมเดล Embedding
    vectors = embedding_model.encode(documents).tolist()  # แปลงข้อความเป็นเวกเตอร์

    # เพิ่มข้อมูลลง Qdrant
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": documents[i]}) for i in range(len(documents))]
    qdrant_client.upsert(collection_name="documents", points=points)

# 6. สร้างฟังก์ชันการค้นหาเอกสาร (ปรับ Limit)
def search_documents(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedding_model.encode([query])[0].tolist()
    search_results = qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=100  # เพิ่ม limit เพื่อให้ครอบคลุมร้านทั้งหมด
    )

    matched_texts = [hit.payload["text"] for hit in search_results]

    return matched_texts

# 7. สร้างฟังก์ชันการสร้างคำตอบ (ปรับปรุงความแม่นยำ)
def generate_answer(query):
    # ค้นหาข้อมูลจาก Qdrant
    retrieved_docs = search_documents(query)

    if not retrieved_docs:
        return "❌ ไม่พบข้อมูลที่คุณต้องการ"

    query_lower = query.lower().strip()  # แปลงเป็นตัวพิมพ์เล็กและลบช่องว่าง

    # ถามเกี่ยวกับอำเภอ
    if "อำเภอ" in query_lower:
        district_name = query_lower.split("อำเภอ")[-1].strip()

        # กรองข้อมูลโดยใช้ regex ที่แม่นยำกว่า
        filtered_docs = [
            doc for doc in retrieved_docs
            if re.search(r"อำเภอ:\s*" + re.escape(district_name), doc.lower())
        ]

        if not filtered_docs:
            return f"📌 ไม่พบข้อมูลร้านคาเฟ่ใน อำเภอ {district_name}"

        # ดึงชื่อร้านจากข้อมูลที่กรองแล้ว
        cafe_names = [
            doc.split("\n")[0].replace("ชื่อร้าน: ", "").strip() for doc in filtered_docs
        ]

        return f"📍 ร้านคาเฟ่ใน อำเภอ {district_name} มีดังนี้:\n" + "\n".join(
            [f"- {name}" for name in cafe_names]
        )

    # ถ้าถามถึงชื่อร้าน → ดึงข้อมูลทั้งหมด
    for doc in retrieved_docs:
        if query_lower in doc.lower():
            return "\n".join(["📌 " + line for line in doc.split("\n")])

    return "❌ ไม่พบข้อมูลร้านคาเฟ่ที่ตรงกับชื่อที่คุณพิมพ์"

# 8. สร้างอินเทอร์เฟซด้วย Streamlit
def main():
    st.title("RAG Chatbot สำหรับข้อมูลคาเฟ่ยอดฮิตใน จังหวัดน่าน")
    st.write("สวัสดี! ฉันคือ Chatbot ที่ช่วยค้นหาข้อมูลคาเฟ่ใน จังหวัดน่าน")

    # ระบุเส้นทางของไฟล์ PDF โดยตรง
    pdf_path = "pdf/คาเฟ่ในอำเภอเมืองจังหวัดน่าน.pdf"

    # อ่านข้อความจากไฟล์ PDF
    documents = prepare_documents_from_pdf(pdf_path)

    # เพิ่มข้อมูลลง Qdrant
    add_documents_to_qdrant(documents)
    st.success("เอกสาร PDF ถูกประมวลผลและพร้อมใช้งานแล้ว!")

    # รับคำถามจากผู้ใช้
    query = st.text_input("คุณ: ", placeholder="พิมพ์คำถามของคุณที่นี่...")

    if st.button("ส่ง"):
        if query:
            # สร้างคำตอบ
            answer = generate_answer(query)
            st.write("Bot:", answer)
        else:
            st.warning("กรุณาพิมพ์คำถามก่อนส่ง")

# 9. เรียกใช้แอปพลิเคชัน
if __name__ == "__main__":
    main()
