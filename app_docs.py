# app.py
import os
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. โหลด environment variables จากไฟล์ .env
load_dotenv()

# 2. เริ่มต้น Qdrant (แบบ In-Memory)
qdrant_client = QdrantClient(":memory:")  # ใช้ In-Memory

# สร้าง Collection สำหรับเก็บเวกเตอร์
qdrant_client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # ใช้ 384-D embedding
)

# 3. ข้อมูลเอกสารที่กำหนดเอง
documents = [
            {
              "name": "Comla Bakery & Baking Studio",
              "address": "59/4 ถนนท่าลี่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
              "phone": "086-925-2626",
              "hours": "ทุกวัน เวลา 08:00 - 18:00 น.",
              "google_maps": "https://www.google.co.th/maps/place/Comla+Bakery/@18.7920485,100.7809916,17z/data=!3m1!4b1!4m14!1m7!3m6!1s0x31278da8c56c7129:0x9addc1127831f954!2sComla+Bakery!8m2!3d18.7920485!4d100.7835665!16s%2Fg%2F11fvl7gl1t!3m5!1s0x31278da8c56c7129:0x9addc1127831f954!8m2!3d18.7920485!4d100.7835665!16s%2Fg%2F11fvl7gl1t?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
              "description": "คาเฟ่ที่มีขนมอบและเค้กหลากหลาย บรรยากาศสบาย ๆ เหมาะสำหรับคนรักการขนมอบ",
              "facebook": "https://www.facebook.com/comlabakery",
              "location": { "lat": 18.7920485, "lng": 100.7809916 }
            },
            {
              "name": "LYKKE Brunch & Wine Bar",
              "address": "160/1 ถนนท่าลี่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
              "phone": "085-697-5935",
              "hours": "ทุกวัน เวลา 09:00 - 22:00 น.",
              "google_maps": "https://www.google.co.th/maps/place/LYKKE+Brunch+%26+Wine+bar/@18.7763039,100.7588063,17z/data=!3m1!4b1!4m6!3m5!1s0x31278f0b9500292d:0x8ac29f400c4b8896!8m2!3d18.7763039!4d100.7588063!16s%2Fg%2F11vhgg1qb3?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
              "description": "คาเฟ่และไวน์บาร์ ที่มีเมนูบรันช์ อาหารเช้า และเครื่องดื่มหลากหลาย",
              "facebook": "https://www.facebook.com/lykkebar",
              "location": { "lat": 18.7763039, "lng": 100.7588063 }
            },
            {
              "name": "Workboxes Café",
              "address": "123/7 ถนนน่าน, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
              "phone": "099-141-2323",
              "hours": "ทุกวัน เวลา 08:00 - 18:00 น.",
              "google_maps": "https://www.google.co.th/maps/place/Workboxes+Cafe'/@18.7883069,100.7756068,17z/data=!3m1!4b1!4m6!3m5!1s0x31278de7c1a3818d:0x12f7e866787be166!8m2!3d18.7883069!4d100.7781817!16s%2Fg%2F11bwkwhgb2?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
              "description": "คาเฟ่ที่มีมุมทำงานสบาย ๆ มีทั้งกาแฟและขนมหวานให้เลือก",
              "facebook": "https://www.facebook.com/workboxes",
              "location": { "lat": 18.7883069, "lng": 100.7756068 }
            },
            {
              "name": "Me & Mum Café",
              "address": "123 ถนนศรีนคร, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
              "phone": "081-625-5353",
              "hours": "ทุกวัน เวลา 08:30 - 17:30 น.",
              "google_maps": "https://www.google.co.th/maps/place/Me+%26+Mum+Caf%C3%A9/@18.8037213,100.7594953,17z/data=!3m1!4b1!4m6!3m5!1s0x31278df4f96ed36f:0xf1b49658f632d34d!8m2!3d18.8037213!4d100.7620702!16s%2Fg%2F11l11cdqlr?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
              "description": "คาเฟ่ขนาดเล็กแต่มีบรรยากาศอบอุ่นและเมนูหลากหลายที่เหมาะกับทุกวัย",
              "facebook": "https://www.facebook.com/meandmum",
              "location": { "lat": 18.8037213, "lng": 100.7594953 }
            },
            {
              "name": "Mix Academic Café",
              "address": "7/2 ถนนหาดใหญ่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
              "phone": "091-234-5678",
              "hours": "ทุกวัน เวลา 09:00 - 19:00 น.",
              "google_maps": "https://www.google.co.th/maps/place/Mix+Academic+Caf%C3%A9/@18.7897171,100.7800158,17z/data=!3m1!4b1!4m6!3m5!1s0x31278de673a42315:0xd7e3dc01a13ac3bc!8m2!3d18.7897171!4d100.7825907!16s%2Fg%2F11bxfvswgy?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
              "description": "คาเฟ่สไตล์การเรียนรู้ ที่มีทั้งหนังสือและเมนูเครื่องดื่มเย็นและร้อน",
              "facebook": "https://www.facebook.com/mixacademic",
              "location": { "lat": 18.7897171, "lng": 100.7800158 }
            },
           {
            "name": "Sober Coffee House",
            "address": "316/4-5 ถนนมหายศ, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "064-496-9199",
            "hours": "ทุกวัน เวลา 08:00 - 17:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/Sober+coffee+house/@18.7788547,100.7712697,17z/data=!3m1!4b1!4m6!3m5!1s0x31278f1cdc82d913:0xbb2ccdd61929aef0!8m2!3d18.7788547!4d100.7738446!16s%2Fg%2F11vyzglpld?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่สไตล์มินิมอลที่มีเมนูเครื่องดื่มหลากหลายและเค้กมะพร้าวที่ได้รับความนิยม",
            "facebook": "https://www.facebook.com/sobercoffeehouse",
            "location": {
              "lat": 18.7788547,
              "lng": 100.7712697
            }
          },
          {
            "name": "Mata Old Town Coffee Nan",
            "address": "45 ถนนสุริยพงษ์, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "089-876-5432",
            "hours": "ทุกวัน เวลา 09:00 - 18:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/%E0%B9%82%E0%B8%A3%E0%B8%87%E0%B8%84%E0%B8%B1%E0%B8%A7%E0%B8%81%E0%B8%B2%E0%B9%81%E0%B8%9F+%E0%B8%A1%E0%B8%B2%E0%B8%95%E0%B8%B2+%E0%B9%82%E0%B8%AD%E0%B8%A5%E0%B8%94%E0%B9%8C+%E0%B8%97%E0%B8%B2%E0%B8%A7%E0%B8%99%E0%B8%8D/@18.7822259,100.7774193,17z/data=!3m1!4b1!4m6!3m5!1s0x31278d001cfd9b73:0x2594b524388a3312!8m2!3d18.7822259!4d100.7799942!16s%2Fg%2F11y62726lx?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่ที่ให้บรรยากาศย้อนยุคสไตล์เมืองเก่าน่าน พร้อมกาแฟคุณภาพดี",
            "facebook": "https://www.facebook.com/mataoldtowncoffee",
            "location": {
              "lat": 18.7822259,
              "lng": 100.7774193
            }
          },
          {
            "name": "Voila Nirvanan (โว้วล่า เนอร์วาน่าน)",
            "address": "82/2 ถนนมหายศ, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "098-765-4321",
            "hours": "ทุกวัน เวลา 07:30 - 17:00 น.",
            "google_maps": "https://www.google.com/maps/place/%E0%B9%82%E0%B8%A3%E0%B8%87%E0%B8%84%E0%B8%B1%E0%B8%A7%E0%B8%81%E0%B8%B2%E0%B9%81%E0%B8%9F+%E0%B8%A1%E0%B8%B2%E0%B8%95%E0%B8%B2+%E0%B9%82%E0%B8%AD%E0%B8%A5%E0%B8%94%E0%B9%8C+%E0%B8%97%E0%B8%B2%E0%B8%A7%E0%B8%99%E0%B8%8D/@18.782231,100.7774193,17z/data=!3m1!4b1!4m6!3m5!1s0x31278d001cfd9b73:0x2594b524388a3312!8m2!3d18.7822259!4d100.7799942!16s%2Fg%2F11y62726lx?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่บรรยากาศอบอุ่น ตกแต่งสไตล์ญี่ปุ่น มีเครื่องดื่มและเบเกอรี่น่าลอง",
            "facebook": "https://www.facebook.com/voilanirvanan",
            "location": {
              "lat": 18.782231,
              "lng": 100.7774193
            }
          },
          {
            "name": "Curve Cafe Nan",
            "address": "99 ถนนน่าน-ท่าวังผา, ตำบลผาสิงห์, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "092-345-6789",
            "hours": "ทุกวัน เวลา 08:00 - 18:00 น.",
            "google_maps": "https://www.google.com/maps/place/CURVE+Coffee+and+Bar/@18.7768559,100.7759484,3a,75y,182.38h,90t/data=!3m7!1e1!3m5!1silD_s_VBGOxuWyq7UNtCew!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D0%26panoid%3DilD_s_VBGOxuWyq7UNtCew%26yaw%3D182.38327!7i16384!8i8192!4m14!1m7!3m6!1s0x31278f4dd513d639:0xc4c5a3d48442d219!2sCURVE+Coffee+and+Bar!8m2!3d18.7767296!4d100.7759501!16s%2Fg%2F11vc2gbvmn!3m5!1s0x31278f4dd513d639:0xc4c5a3d48442d219!8m2!3d18.7767296!4d100.7759501!16s%2Fg%2F11vc2gbvmn?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่บรรยากาศโมเดิร์น พร้อมกาแฟคุณภาพและวิวธรรมชาติ",
            "facebook": "https://www.facebook.com/curvecafenan",
            "location": {
              "lat": 18.7768559,
              "lng": 100.7759484
            }
          },
          {
            "name": "น.น่าน คาเฟ่",
            "address": "ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "093-456-7890",
            "hours": "ทุกวัน เวลา 08:00 - 17:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/%E0%B8%99.%E0%B8%99%E0%B9%88%E0%B8%B2%E0%B8%99+%E0%B8%84%E0%B8%B2%E0%B9%80%E0%B8%9F%E0%B9%88/@18.7892887,100.7841151,17z/data=!3m1!4b1!4m6!3m5!1s0x31278e7679e33df5:0x2efa548613c1025a!8m2!3d18.7892887!4d100.78669!16s%2Fg%2F11bxgmx_p1?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่บรรยากาศสงบ เหมาะสำหรับการพักผ่อนและชิลล์กับเครื่องดื่มและขนม",
            "facebook": "https://www.facebook.com/nan.cafe",
            "location": {
              "lat": 18.7892887,
              "lng": 100.7841151
            }
          },
           {
            "name": "Inlamai Coffee",
            "address": "150 ถนนท่าลี่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "098-123-4567",
            "hours": "ทุกวัน เวลา 08:00 - 18:00 น.",
            "google_maps": "https://www.google.com/maps/place/inlamai+coffee/@18.7828238,100.7695158,17z/data=!4m15!1m8!3m7!1s0x30dddf3e4f732c13:0x2a65a6f734774880!2sinlamai+coffee!8m2!3d18.7828238!4d100.7720907!10e1!16s%2Fg%2F11sgp1pjlm!3m5!1s0x30dddf3e4f732c13:0x2a65a6f734774880!8m2!3d18.7828238!4d100.7720907!16s%2Fg%2F11sgp1pjlm?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีการเสิร์ฟกาแฟคุณภาพดีและบรรยากาศสบาย ๆ เหมาะสำหรับการนั่งทำงานหรือพบปะเพื่อนฝูง",
            "facebook": "https://www.facebook.com/inlamaicoffee",
            "location": { "lat": 18.7828238, "lng": 100.7695158 }
          },
          {
            "name": "Coffee Room Nan",
            "address": "78 ถนนพหลโยธิน, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "095-678-1234",
            "hours": "ทุกวัน เวลา 09:00 - 18:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/Coffee+Room+Nan/@18.7698788,100.7628357,17z/data=!3m1!4b1!4m6!3m5!1s0x31278fef76f93dd5:0x272e7626267c791e!8m2!3d18.7698788!4d100.7654106!16s%2Fg%2F11sv2z4k1g?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีเมนูกาแฟและขนมอบหลายชนิด พร้อมบรรยากาศที่เงียบสงบ",
            "facebook": "https://www.facebook.com/coffeeroomnan",
            "location": { "lat": 18.7698788, "lng": 100.7628357 }
          },
          {
            "name": "THE CORE COFFEEBAR",
            "address": "45 ถนนหาดใหญ่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "091-234-5678",
            "hours": "ทุกวัน เวลา 08:00 - 19:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/THE+CORE+COFFEEBAR/@18.7801302,100.7751906,17z/data=!3m1!4b1!4m6!3m5!1s0x31278d7b2ca9b4a5:0x6b6f0c8652df8bb2!8m2!3d18.7801302!4d100.7777655!16s%2Fg%2F11rg7s37yv?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่ที่เน้นกาแฟและเครื่องดื่มที่มีรสชาติกลมกล่อม พร้อมมุมพักผ่อนที่สะดวกสบาย",
            "facebook": "https://www.facebook.com/thecorecoffeebar",
            "location": { "lat": 18.7801302, "lng": 100.7751906 }
          },
          {
            "name": "ลืมเวลา คาเฟ่ (Luem Wela Cafe)",
            "address": "25/2 ถนนมหายศ, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "092-345-6789",
            "hours": "ทุกวัน เวลา 09:00 - 18:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/%E0%B8%A5%E0%B8%B7%E0%B8%A1%E0%B9%80%E0%B8%A7%E0%B8%A5%E0%B8%B2+%E0%B8%84%E0%B8%B2%E0%B8%9F%E0%B9%88+(Luem+Wela+Cafe)/@18.8121285,100.7599702,17z/data=!3m1!4b1!4m6!3m5!1s0x31278d5b80c8c947:0xbb8f8185c9b314af!8m2!3d18.8121285!4d100.7625451!16s%2Fg%2F11tf3nf7cq?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่บรรยากาศเหมาะสำหรับการหลีกหนีความวุ่นวาย มีเมนูกาแฟหลากหลาย",
            "facebook": "https://www.facebook.com/luemwelacafe",
            "location": { "lat": 18.8121285, "lng": 100.7599702 }
          },
          {
            "name": "SommePaul Cafe",
            "address": "101/5 ถนนท่าลี่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "089-456-7890",
            "hours": "ทุกวัน เวลา 08:00 - 17:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/SommePaul+cafe/@18.773907,100.7645429,17z/data=!3m1!4b1!4m6!3m5!1s0x31278f7cef3b8c6d:0x1c90dc9f76180d6d!8m2!3d18.773907!4d100.7671178!16s%2Fg%2F11kj1gktg7?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีเมนูอาหารและเครื่องดื่มหลากหลาย พร้อมบรรยากาศที่น่านั่ง",
            "facebook": "https://www.facebook.com/sommePaulcafe",
            "location": { "lat": 18.773907, "lng": 100.7645429 }
          },
          {
            "name": "Southern Coffee",
            "address": "35 ถนนท่าใหม่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "097-654-3210",
            "hours": "ทุกวัน เวลา 09:00 - 18:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/Southern+coffee/@18.7779595,100.755453,17z/data=!3m1!4b1!4m6!3m5!1s0x31278f274c8a3aad:0xaf79d2f8254f5e23!8m2!3d18.7779595!4d100.7580279!16s%2Fg%2F11fvl51p9m?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่ที่เสิร์ฟกาแฟรสชาติเยี่ยม พร้อมเมนูขนมเบเกอรี่หลากหลาย",
            "facebook": "https://www.facebook.com/southerncoffee",
            "location": { "lat": 18.7779595, "lng": 100.755453 }
          },
          {
            "name": "DRIP IN HOME Cafe’ Nan",
            "address": "72/3 ถนนสุริยพงษ์, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "095-567-1234",
            "hours": "ทุกวัน เวลา 08:30 - 18:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/DRIP+IN+HOME+Cafe%E2%80%99+Nan/@18.7385059,100.7527157,17z/data=!3m1!4b1!4m6!3m5!1s0x31278fdf785d4053:0x755807533c224aaa!8m2!3d18.7385059!4d100.7552906!16s%2Fg%2F11svl70xk5?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีทั้งกาแฟดริปและเมนูเครื่องดื่มสุดพิเศษ พร้อมบรรยากาศผ่อนคลาย",
            "facebook": "https://www.facebook.com/dripinhomecafe",
            "location": { "lat": 18.7385059, "lng": 100.7527157 }
          },
          {
            "name": "Namwan Cafe",
            "address": "16 ถนนหาดใหญ่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "089-567-4321",
            "hours": "ทุกวัน เวลา 09:00 - 17:00 น.",
            "google_maps": "https://www.google.co.th/maps/place/Namwan+Cafe/@18.7749375,100.7593626,17z/data=!3m1!4b1!4m6!3m5!1s0x31278fe921ef5d95:0x58ec99877371941c!8m2!3d18.7749375!4d100.7619375!16s%2Fg%2F11t8b3s_px?hl=th&entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoJLDEwMjExNDUzSAFQAw%3D%3D",
            "description": "คาเฟ่ที่เน้นเครื่องดื่มเย็นและขนมท้องถิ่น พร้อมบรรยากาศอบอุ่น",
            "facebook": "https://www.facebook.com/namwancafe",
            "location": { "lat": 18.7749375, "lng": 100.7593626 }
          },
          {
            "name": "la Mure Cafe ละเมอ คาเฟ่",
            "address": "89 ถนนท่าใหม่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "094-567-8901",
            "hours": "ทุกวัน เวลา 08:00 - 18:00 น.",
            "google_maps": "https://www.google.com/maps/place/la+Mure+cafe+%E0%B8%A5%E0%B8%B0%E0%B9%80%E0%B8%A1%E0%B8%AD+%E0%B8%84%E0%B8%B2%E0%B9%80%E0%B8%9F%E0%B9%88/@18.8443711,100.7764111,17z/data=!3m2!4b1!5s0x30dddf6e5c431441:0xa81e7027284c1a5a!8m2!3d18.8443711!4d100.7764111",
            "description": "ร้านกาแฟที่เน้นกาแฟดริปและขนมหวานท้องถิ่น พร้อมบรรยากาศน่ารัก",
            "facebook": "https://www.facebook.com/lamurecafe",
            "location": { "lat": 18.8443711, "lng": 100.7764111 }
          },
           {
            "name": "น่านวนา NANVANA",
            "address": "10/1 ถนนมหายศ, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "083-456-1234",
            "hours": "ทุกวัน เวลา 08:00 - 17:00 น.",
            "google_maps": "https://www.google.com/maps/place/%E0%B8%99%E0%B9%88%E0%B8%B2%E0%B8%99%E0%B8%A7%E0%B8%99%E0%B8%B2+NANVANA/@18.7745895,100.748723,17z/data=!3m1!4b1!4m6!3m5!1s0x31278ffd843a5241:0xdbd6869c2bfebb46!8m2!3d18.7745844!4d100.7512979!16s%2Fg%2F11rqtxplxy?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีบรรยากาศสบาย ๆ และเครื่องดื่มรสชาติพิเศษ",
            "facebook": "https://www.facebook.com/nanvana",
            "location": { "lat": 18.7745895, "lng": 100.748723 }
          },
          {
            "name": "ลิลินน์ คาเฟ่",
            "address": "112/3 ถนนสุริยพงษ์, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "086-123-4567",
            "hours": "ทุกวัน เวลา 09:00 - 18:00 น.",
            "google_maps": "https://www.google.com/maps/search/%E0%B8%A5%E0%B8%B4%E0%B8%A5%E0%B8%B4%E0%B8%99%E0%B8%99%E0%B9%8C+%E0%B8%84%E0%B8%B2%E0%B9%80%E0%B8%9F%E0%B9%88/@18.7715764,100.5126426,10z/data=!3m1!4b1?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีมุมผ่อนคลายและกาแฟสด",
            "facebook": "https://www.facebook.com/lilinndcafe",
            "location": { "lat": 18.7715764, "lng": 100.5126426 }
          },
          {
            "name": "Blur Cafe Coffee & Cozy Space",
            "address": "143 ถนนท่าลี่, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "091-234-6789",
            "hours": "ทุกวัน เวลา 08:00 - 18:00 น.",
            "google_maps": "https://www.google.com/maps/place/Blur+cafe+coffee%26Cozyspace/@18.7749212,100.7590936,17z/data=!3m1!4b1!4m6!3m5!1s0x31278f65c92cb627:0xcde191296a686754!8m2!3d18.7749161!4d100.7616685!16s%2Fg%2F11q4j4cvp1?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีมุมพักผ่อนและกาแฟรสเข้ม",
            "facebook": "https://www.facebook.com/blurcozyspace",
            "location": { "lat": 18.7749212, "lng": 100.7590936 }
          },
          {
            "name": "บ้านนาก๋างโต้ง Baan Na Kang Tong Cafe & Homestay",
            "address": "27/1 ถนนน่าน-ท่าวังผา, ตำบลผาสิงห์, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "085-123-6789",
            "hours": "ทุกวัน เวลา 09:00 - 18:00 น.",
            "google_maps": "https://www.google.com/maps/place/%E0%B8%9A%E0%B9%89%E0%B8%B2%E0%B8%99%E0%B8%99%E0%B8%B2%E0%B8%81%E0%B9%8B%E0%B8%B2%E0%B8%87%E0%B9%82%E0%B8%95%E0%B9%89%E0%B8%87+Baan+Na+Kang+Tong+Cafe+%26+Homestay/@18.8136957,100.729945,17z/data=!4m9!3m8!1s0x31278da231058d37:0xe0d77fc6544860ba!5m2!4m1!1i2!8m2!3d18.8136906!4d100.7325199!16s%2Fg%2F11knnlfqs_?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่และโฮมสเตย์ที่มีบรรยากาศธรรมชาติ พร้อมเมนูเครื่องดื่มและอาหาร",
            "facebook": "https://www.facebook.com/baankangtongcafe",
            "location": { "lat": 18.8136957, "lng": 100.729945 }
          },
          {
            "name": "เอราบิก้า คอฟฟี น่าน",
            "address": "60 ถนนมหายศ, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "064-123-4567",
            "hours": "ทุกวัน เวลา 08:00 - 17:00 น.",
            "google_maps": "https://www.google.com/maps/place/%E0%B9%80%E0%B8%AD%E0%B8%A3%E0%B8%B2%E0%B8%9A%E0%B8%B4%E0%B8%81%E0%B9%89%E0%B8%B2+%E0%B8%84%E0%B8%AD%E0%B8%9F%E0%B8%9F%E0%B8%B5+%E0%B8%99%E0%B9%88%E0%B8%B2%E0%B8%99/@18.787195,100.7816768,17z/data=!3m1!4b1!4m6!3m5!1s0x31278de43bef1f19:0x710d375d31197b82!8m2!3d18.7871899!4d100.7842517!16s%2Fg%2F11b6gm1335?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีเมนูกาแฟสดและขนมเค้กสูตรพิเศษ",
            "facebook": "https://www.facebook.com/arabicacoffee",
            "location": { "lat": 18.787195, "lng": 100.7816768 }
          },
          {
            "name": "All Era Cafe’",
            "address": "123 ถนนสุริยพงษ์, ตำบลในเวียง, อำเภอเมืองน่าน, จังหวัดน่าน 55000",
            "phone": "089-765-4321",
            "hours": "ทุกวัน เวลา 08:00 - 18:00 น.",
            "google_maps": "https://www.google.com/maps/place/All+Era+Cafe%E2%80%99/@18.8077428,100.771273,17z/data=!3m1!4b1!4m6!3m5!1s0x31278d3285ea03a9:0xe49dcc5f530fec4b!8m2!3d18.8077377!4d100.7738479!16s%2Fg%2F11trtkcg3b?entry=ttu&g_ep=EgoyMDI1MDIyMy4xIKXMDSoASAFQAw%3D%3D",
            "description": "คาเฟ่ที่มีทั้งเครื่องดื่มและขนม พร้อมบรรยากาศสะดวกสบาย",
            "facebook": "https://www.facebook.com/alleracafe",
            "location": { "lat": 18.8077428, "lng": 100.771273 }
          }
      ]

return additional_documents

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
    st.title("RAG Chatbot คาเฟ่ในอำเภอเมือง จังหวัดน่าน")
    st.write("สวัสดี Chatbot ที่ช่วยตอบคำถามจากเอกสารที่มีอยู่")

    # เพิ่มข้อมูลเอกสารลงใน Qdrant
    add_documents_to_qdrant(documents)
    st.success("ข้อมูลเอกสารพร้อมใช้งานแล้ว!")

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