import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re

# ----------------- الإعدادات -----------------
TARGET_URL = "https://www.football-data.co.uk/englandm.php"
BASE_URL = "https://www.football-data.co.uk/"
DOWNLOAD_DIR = "historical_data"
MIN_SEASON_YEAR = 14  # تجاهل ما قبل موسم 2014/2015

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
}

# 🧠 الترقية 1: نمط ذكي (Regex) لاصطياد روابط الدوري الإنجليزي والمواسم بدقة
E0_PATTERN = re.compile(r'mmz4281/(\d{4})/E0\.csv', re.IGNORECASE)

def download_premier_league_data():
    print(f"⏳ جاري الاتصال بموقع Football-Data.co.uk...")
    
    try:
        response = requests.get(TARGET_URL, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        
        csv_files = []
        download_count = 0
        skip_count = 0
        
        print("🔍 جاري الفحص الذكي للروابط...")
        
        for link in links:
            href = link.get('href')
            if not href:
                continue
                
            # محاولة مطابقة الرابط مع النمط الذكي
            match = E0_PATTERN.search(href)
            if match:
                season_folder = match.group(1) # سيستخرج '2324' أو '1415'
                
                # تصفية المواسم القديمة
                try:
                    start_year = int(season_folder[:2])
                    if start_year < MIN_SEASON_YEAR and start_year != 0:
                        continue
                except ValueError:
                    continue
                    
                full_url = urljoin(BASE_URL, href)
                file_name = f"E0_{season_folder}.csv"
                file_path = os.path.join(DOWNLOAD_DIR, file_name)
                
                csv_files.append(file_path)
                
                # 🧠 الترقية 2: الذاكرة الذكية (تخطي الملفات الموجودة)
                if os.path.exists(file_path):
                    skip_count += 1
                    continue
                
                print(f"📥 جاري تحميل موسم {season_folder[:2]}/{season_folder[2:]} ...")
                csv_response = requests.get(full_url, headers=HEADERS)
                
                if csv_response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(csv_response.content)
                    download_count += 1
                    time.sleep(0.5) # راحة قصيرة لتجنب الحظر
                else:
                    print(f"⚠️ فشل تحميل الرابط: {full_url}")
        
        print(f"\n✅ النتيجة: تم تحميل {download_count} ملف جديد، وتخطي {skip_count} ملف موجود مسبقاً.")
        return csv_files
        
    except Exception as e:
        print(f"❌ حدث خطأ أثناء التحميل: {e}")
        return []

def smart_read_csv(file_path):
    """🧠 الترقية 3: قارئ ذكي يجرب ترميزات مختلفة لتفادي أخطاء الملفات"""
    encodings = ['utf-8', 'windows-1252', 'latin1', 'unicode_escape']
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            # 🧠 الترقية 4: تنظيف المسافات المخفية في أسماء الأعمدة
            df.columns = df.columns.str.strip()
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
            
    print(f"⚠️ فشل في قراءة الملف بجميع الترميزات: {file_path}")
    return None

def merge_csv_files(file_list, output_name="Master_E0_Database.csv"):
    print("\n🔄 جاري دمج كل المواسم في قاعدة بيانات واحدة...")
    
    all_dataframes = []
    
    # الأعمدة الذهبية التي يحتاجها الذكاء الاصطناعي (تم تنظيف مسافاتها)
    important_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                      'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 
                      'HY', 'AY', 'HR', 'AR']
    
    for file in file_list:
        df = smart_read_csv(file)
        
        if df is not None:
            # فلترة الأعمدة الموجودة فعلياً في الملف (تحسباً لاختلاف الملفات القديمة)
            available_cols = [col for col in important_cols if col in df.columns]
            df_clean = df[available_cols].copy()
            
            # إزالة الصفوف الفارغة أو المباريات المؤجلة
            df_clean = df_clean.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG'])
            
            if not df_clean.empty:
                all_dataframes.append(df_clean)
            
    if all_dataframes:
        master_df = pd.concat(all_dataframes, ignore_index=True)
        master_df.to_csv(output_name, index=False)
        print(f"🌟 تم الانتهاء بنجاح! قاعدة البيانات المدمجة: {output_name}")
        print(f"📊 إجمالي عدد المباريات الجاهزة لتدريب الذكاء الاصطناعي: {len(master_df)} مباراة!")
    else:
        print("❌ لم يتم العثور على بيانات صالحة لدمجها.")

# ----------------- التشغيل -----------------
if __name__ == "__main__":
    downloaded_files = download_premier_league_data()
    if downloaded_files:
        # ترتيب الملفات زمنياً ليكون الدمج صحيحاً
        downloaded_files.sort()
        merge_csv_files(downloaded_files, "E0_Master.csv")
