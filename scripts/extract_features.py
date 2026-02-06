import json
import pandas as pd
import re
import os

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_features(data):
    
    # 1. Attendance
    # حضور مهم است و تاثیر مستقیم دارد -> strict
    # حضور مهم نیست اما تاثیر مثبت دارد -> bonus
    # حضور و غیاب نمی کند -> optional
    def map_attendance(text):
        if not text: return "unknown"
        if "مهم است" in text or "تاثیر مستقیم" in text:
            return "strict"
        if "مهم نیست" in text or "تاثیر مثبت" in text:
            return "bonus"
        if "نمی کند" in text or "نمیکند" in text:
            return "optional"
        return "unknown"

    # 2. Exam Adequacy
    # بله -> yes
    # تا حدودی -> partial
    # خیر -> no
    def map_exam_adequacy(text):
        if not text: return "unknown"
        if "بله" in text:
            return "yes"
        if "تا حدودی" in text or "تاحدودی" in text:
            return "partial"
        if "خیر" in text or "اصلا" in text:
            return "no"
        return "unknown"

    # 3. Grading
    # منصفانه و هرچی خودت بگیری -> fair
    # دست باز و با ارفاق -> lenient
    # نمره خوبی نمیشه ازشون گرفت -> strict
    def map_grading(text):
        if not text: return "unknown"
        if "منصفانه" in text or "هرچی خودت" in text:
            return "fair"
        if "دست باز" in text or "ارفاق" in text:
            return "lenient"
        if "نمره خوبی نمیشه" in text or "خوب نمیده" in text or "بد نمره" in text:
            return "strict"
        return "unknown"
        
    # 4. Teaching Resources (Multi-tag potential, but simplified for now)
    # فایل پاورپوینت... -> slides
    # کتاب مرجع -> textbook
    # جزوه می نویسیم -> handwritten_notes
    # جزوات ترم های گذشته -> old_notes
    # منبعی معرفی نمی کنند -> none
    def map_resources(text):
        if not text: return "unknown"
        if "پاورپوینت" in text or "پی دی اف" in text:
            return "slides"
        if "کتاب مرجع" in text or "کتاب" in text:
            return "textbook"
        if "می نویسیم" in text or "مینویسیم" in text:
            return "handwritten_notes"
        if "ترم های گذشته" in text:
            return "old_notes"
        if "معرفی نمی کنند" in text or "منبعی" in text:
            return "none"
        return "other"

    processed_count = 0
    for entry in data:
        entry['feature_attendance'] = map_attendance(entry.get('attendance'))
        entry['feature_exam_adequacy'] = map_exam_adequacy(entry.get('exam_adequacy'))
        entry['feature_grading'] = map_grading(entry.get('grading'))
        entry['feature_resources'] = map_resources(entry.get('teaching_resources'))
        processed_count += 1
        
    return data

def main():
    input_file = os.path.join('data', 'processed', 'normalized_comments.json')
    output_file = os.path.join('data', 'processed', 'normalized_comments.json') 
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    print("Loading data...")
    data = load_data(input_file)
    
    print("Extracting features...")
    data = extract_features(data)
    
    print(f"Saving enriched data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("Done! Added features: feature_attendance, feature_exam_adequacy, feature_grading, feature_resources")
    
    # Print sample to verify
    print("\nSample Entry:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
