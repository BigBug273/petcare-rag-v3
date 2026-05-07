import os
import re
import time
import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ==========================================
# Clean Text
# ==========================================
def clean_data(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def wait_page_load(driver, timeout=10):
    """รอให้หน้าเว็บโหลด body ก่อนค่อย scrape"""
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )


def is_valid_paragraph(text):
    """กรอง paragraph ที่เป็นขยะออก"""
    if not text:
        return False

    if len(text) < 30:
        return False

    junk_keywords = [
        "คุกกี้",
        "cookie",
        "privacy",
        "terms",
        "ติดตามเรา",
        "สมัครรับข่าวสาร",
        "©",
    ]

    lower_text = text.lower()
    for keyword in junk_keywords:
        if keyword.lower() in lower_text:
            return False

    return True


# ==========================================
# Scraping Function
# ==========================================
def start_scraping_selenium():
    print("🤖 ระบบกำลังเตรียมเปิด Microsoft Edge จำลอง...")

    edge_options = Options()
    # edge_options.add_argument("--headless")  # ถ้าอยากซ่อนหน้าต่าง ให้เอา # ออก

    driver = webdriver.Edge(options=edge_options)

    targets = [
        {
            "type": "cat",
            "urls": [
                "https://www.purina.co.th/find-a-pet/cat/breed-library?page=0",
                "https://www.purina.co.th/find-a-pet/cat/breed-library?page=1",
            ],
        },
        {
            "type": "dog",
            "urls": [
                "https://www.purina.co.th/find-a-pet/dog/breed-library?page=0",
                "https://www.purina.co.th/find-a-pet/dog/breed-library?page=1",
                "https://www.purina.co.th/find-a-pet/dog/breed-library?page=2",
            ],
        },
    ]

    results = []
    visited_urls = set()

    try:
        for target in targets:
            pet_type = target["type"]

            for url in target["urls"]:
                print(f"\n🔍 กำลังเปิดหน้าเว็บ {pet_type}: {url}")
                driver.get(url)
                wait_page_load(driver)
                time.sleep(2)

                soup = BeautifulSoup(driver.page_source, "html.parser")

                all_links = soup.find_all("a", href=True)
                breed_links = []

                for a in all_links:
                    href = a["href"]
                    href_clean = href.split("?")[0]

                    # หาเฉพาะ link รายละเอียดสายพันธุ์
                    if (
                        f"/{pet_type}/" in href_clean
                        and href_clean != f"/find-a-pet/{pet_type}/breed-library"
                    ):
                        full_url = (
                            "https://www.purina.co.th" + href
                            if href.startswith("/")
                            else href
                        )

                        if (
                            "purina.co.th" in full_url
                            and full_url not in breed_links
                        ):
                            breed_links.append(full_url)

                print(f"✅ พบ {len(breed_links)} สายพันธุ์ในหน้านี้")

                for detail_url in breed_links:
                    if detail_url in visited_urls:
                        continue

                    visited_urls.add(detail_url)

                    print(f"   -> กำลังเข้าหน้า: {detail_url}")
                    driver.get(detail_url)
                    wait_page_load(driver)
                    time.sleep(1.5)

                    detail_soup = BeautifulSoup(driver.page_source, "html.parser")

                    # ดึงชื่อสายพันธุ์
                    h1_tag = detail_soup.find("h1")
                    breed_name = (
                        h1_tag.get_text(" ", strip=True)
                        if h1_tag
                        else detail_url.split("/")[-1].replace("-", " ")
                    )
                    breed_name = clean_data(breed_name)

                    # ดึงเฉพาะ content หลัก ถ้าเจอ article/main ให้ใช้ก่อน
                    main_content = (
                        detail_soup.find("article")
                        or detail_soup.find("main")
                        or detail_soup
                    )

                    paragraphs = main_content.find_all("p")
                    paragraph_texts = []

                    for p in paragraphs:
                        txt = clean_data(p.get_text(" ", strip=True))

                        if is_valid_paragraph(txt):
                            paragraph_texts.append(txt)

                    full_text = clean_data(" ".join(paragraph_texts))

                    # กันกรณีข้อมูลยาวเกินไปสำหรับ demo / embedding
                    full_text = full_text[:3000]

                    print(f"      📄 {breed_name} | text length: {len(full_text)}")

                    # ถ้าหน้านั้นไม่มีข้อมูลจริง ให้ข้าม
                    if len(full_text) < 50:
                        print("      ⚠️ ข้าม เพราะข้อมูลสั้นเกินไป")
                        continue

                    data = {
                        "type": pet_type,
                        "breed_name": breed_name,
                        "full_text": full_text,
                        "source_url": detail_url,
                    }

                    results.append(data)

    finally:
        driver.quit()

    return results


# ==========================================
# Save CSV
# ==========================================
if __name__ == "__main__":
    print("🚀 เริ่มกระบวนการดึงข้อมูลด้วย Selenium...")

    final_data = start_scraping_selenium()

    columns = [
        "type",
        "breed_name",
        "full_text",
        "source_url",
    ]

    if len(final_data) > 0:
        df = pd.DataFrame(final_data)
        df = df[columns]

        # ลบข้อมูลซ้ำจาก source_url
        df = df.drop_duplicates(subset=["source_url"])

        os.makedirs("data", exist_ok=True)

        # save เป็นไฟล์หลัก
        output_path = "data/pet_breeds.csv"

        try:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        except PermissionError:
            # ถ้าเปิดไฟล์ CSV ค้างใน Excel/VS Code จะ save ไฟล์หลักไม่ได้
            output_path = "data/pet_breeds_new.csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print("⚠️ ไฟล์ pet_breeds.csv ถูกเปิดค้างอยู่ เลยบันทึกเป็น pet_breeds_new.csv แทน")

        print(f"\n🎉 สำเร็จแล้ว! ดึงข้อมูลมาได้ทั้งหมด {len(df)} สายพันธุ์")
        print(f"📁 บันทึกไฟล์เรียบร้อยที่: {output_path}")
        print("\nตัวอย่างข้อมูล:")
        print(df.head())
    else:
        print("\n⚠️ ยังไม่ได้ข้อมูล ต้องเช็กโครงสร้างหน้าเว็บอีกที")