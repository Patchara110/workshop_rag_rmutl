def main():
    st.title("RAG Chatbot เกี่ยวกับคาเฟ่ในตัวอำเภอเมือง จังหวัดน่าน")
    st.write("สวัสดี Chatbot ที่ช่วยตอบคำถามเกี่ยวกับคาเฟ่ในตัวอำเภอเมือง จังหวัดน่าน")

    # กำหนด path ของไฟล์ PDF
    pdf_path = "pdf/คาเฟ่ในอำเภอเมืองจังหวัดน่าน.pdf"  # เปลี่ยน path ของไฟล์ PDF ให้ตรงกับไฟล์ที่มีข้อมูลคาเฟ่

    # ตรวจสอบว่าไฟล์ PDF มีอยู่
    if os.path.exists(pdf_path):
        st.write("ไฟล์ PDF พบ")
        # อ่านข้อความจากไฟล์ PDF
        documents = prepare_documents_from_pdf(pdf_path)

        # เพิ่มข้อมูลลง Qdrant
        add_documents_to_qdrant(documents)
        st.success("เอกสาร PDF เกี่ยวกับคาเฟ่ในตัวอำเภอเมือง จังหวัดน่าน ถูกประมวลผลและพร้อมใช้งานแล้ว!")

        # รับคำถามจากผู้ใช้
        query = st.text_input("คุณ: ", placeholder="พิมพ์คำถามเกี่ยวกับคาเฟ่ที่นี่...")

        st.write(f"คำถามที่พิมพ์: {query}")  # แสดงคำถามที่กรอกเข้ามา

        if st.button("ส่ง"):
            if query:
                # สร้างคำตอบ
                answer = generate_answer(query)
                st.write("Bot:", answer)
            else:
                st.warning("กรุณาพิมพ์คำถามก่อนส่ง")
    else:
        st.error(f"ไม่พบไฟล์ PDF ที่ path: {pdf_path}")
        st.stop()