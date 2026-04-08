import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd

# رابط الصفحة التي تحتوي على الروابط
TARGET_URL = "https://www.football-data.co.uk/englandm.php"
BASE_URL = "https://www.football-data.co.uk/"

# مجلد لحفظ الملفات المحملة
DOWNLOAD_DIR = "historical_data"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# نتظاهر بأننا متصفح عادي احتراما للموقع
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
}

def download_premier_league_data():
    print(f"⏳ جاري الاتصال بموقع Football-Data.co.uk...")
    
    try:
        response = requests.get(TARGET_URL, headers=HEADERS)
        response.raise_for_status()
        
        # تحليل كود الـ HTML للصفحة
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # البحث عن كل الروابط <a> في الصفحة
        links = soup.find_all('a')
        
        csv_files = []
        download_count = 0
        
        print("🔍 جاري البحث عن ملفات الدوري الإنجليزي الممتاز (E0.csv)...")
        
        for link in links:
            href = link.get('href')
            
            # نحن نبحث فقط عن الروابط التي تنتهي بـ E0.csv (الدوري الممتاز)
            if href and href.endswith('E0.csv'):
                # استخراج اسم الموسم من الرابط (مثلا 2324 تعني موسم 23/24)
                # الرابط يكون عادة هكذا: mmz4281/2324/E0.csv
                parts = href.split('/')
                if len(parts) >= 3:
                    season_folder = parts[1] # مثل '2324'
                    
                    # لا نريد مواسم قديمة جداً (مثلا قبل 2014) لأن التكتيكات تغيرت
                    # يمكنك إلغاء هذا الشرط إذا أردت كل المواسم منذ 1993!
                    try:
                        start_year = int(season_folder[:2])
                        if start_year < 14 and start_year > 0: # تجاهل ما قبل موسم 14/15
                            continue
                    except:
                        pass
                        
                    full_url = urljoin(BASE_URL, href)
                    file_name = f"E0_{season_folder}.csv"
                    file_path = os.path.join(DOWNLOAD_DIR, file_name)
                    
                    # تحميل الملف
                    print(f"📥 جاري تحميل موسم {season_folder[:2]}/{season_folder[2:]} ...")
                    csv_response = requests.get(full_url, headers=HEADERS)
                    
                    with open(file_path, 'wb') as f:
                        f.write(csv_response.content)
                    
                    csv_files.append(file_path)
                    download_count += 1
                    
                    # توقف لثانية واحدة احتراماً لخوادم الموقع (Polite Scraping)
                    time.sleep(1)
        
        print(f"\n✅ تم تحميل {download_count} مواسم بنجاح!")
        return csv_files
        
    except Exception as e:
        print(f"❌ حدث خطأ أثناء التحميل: {e}")
        return []

def merge_csv_files(file_list, output_name="Master_E0_Database.csv"):
    """دمج كل المواسم المحملة في قاعدة بيانات واحدة ضخمة للذكاء الاصطناعي"""
    print("\n🔄 جاري دمج كل المواسم في ملف واحد ضخم...")
    
    all_dataframes = []
    
    for file in file_list:
        try:
            # قراءة الملف (بعض الملفات القديمة قد تحتوي أخطاء ترميز، لذا نستخدم unicode_escape أو latin1)
            df = pd.read_csv(file, encoding='unicode_escape')
            
            # نحتفظ فقط بالأعمدة التي تهم نموذجنا (لتنظيف البيانات)
            important_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                              'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 
                              'HY', 'AY', 'HR', 'AR']
            
            # التأكد من أن الأعمدة موجودة في الملف
            available_cols = [col for col in important_cols if col in df.columns]
            df_clean = df[available_cols]
            
            # حذف الصفوف الفارغة
            df_clean = df_clean.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG'])
            
            all_dataframes.append(df_clean)
        except Exception as e:
            print(f"⚠️ تحذير: مشكلة في قراءة ملف {file} - {e}")
            
    if all_dataframes:
        # دمج كل الجداول
        master_df = pd.concat(all_dataframes, ignore_index=True)
        
        # حفظ الملف النهائي بجوار السكريبت الخاص بك
        master_df.to_csv(output_name, index=False)
        print(f"🌟 تم الانتهاء! تم إنشاء قاعدة البيانات: {output_name}")
        print(f"📊 إجمالي عدد المباريات الجاهزة لتدريب الذكاء الاصطناعي: {len(master_df)} مباراة!")
    else:
        print("❌ لم يتم العثور على بيانات لدمجها.")

# --- التشغيل ---
if __name__ == "__main__":
    downloaded_files = download_premier_league_data()
    if downloaded_files:
        merge_csv_files(downloaded_files, "E0_Master.csv")
