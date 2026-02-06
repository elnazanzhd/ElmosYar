import json
import re
import unicodedata
import os

def normalize_text(text):
    if not text:
        return ""
    
    # 1. Unicode normalization: NFKC
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Arabic -> Persian letter mapping
    text = text.replace('ي', 'ی')
    text = text.replace('ك', 'ک')
    text = text.replace('ة', 'ه')
    text = re.sub(r'[أإؤ]', 'ا', text)
    
    # 3. Digits normalization
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"
    
    trans_table = str.maketrans(persian_digits + arabic_digits, english_digits * 2)
    text = text.translate(trans_table)
    

    word_num_map = {
        'یک': '1',
        'دو': '2',
        'سه': '3',
        'چهار': '4',

    }
    
    for word, digit in word_num_map.items():
        # Replace if word is surrounded by whitespace or start/end of string
        # pattern: (?<=\s|^)word(?=\s|$)
  
        text = re.sub(r'(?<=[\s\u200c])' + word + r'(?=[\s\u200c]|$)', digit, text)
        text = re.sub(r'^' + word + r'(?=[\s\u200c]|$)', digit, text)


    text = text.replace('\u200c', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Punctuation normalization
    # Unify colons
    text = text.replace(':', ':')
    text = re.sub(r'[:：]', ':', text)
    # Unify dashes
    text = re.sub(r'[-－—–]', '-', text)
    return text

def normalize_professor_name(name):
    if not name:
        return ""
    
    name = normalize_text(name)
    
    # Preserve anonymous names (e.g., "استاد 123")
    if re.match(r'^استاد \d+$', name):
        return name

    titles = ["دکتر", "مهندس", "استاد"]
    titles.sort(key=len, reverse=True)
    
    found_title = True
    while found_title:
        found_title = False
        for title in titles:
            
            if name.startswith(title):
                
                if name.startswith(title + " "):
                    name = name[len(title)+1:].strip()
                    found_title = True
                    break 
                elif name == title:
                    name = ""
                    found_title = True
                    break
                
                pass
                
    return name

def normalize_faculty_name(name):
    if not name:
        return ""
    
   
    name = normalize_text(name)

    
    prefixes = ["مهندسی ", "مهندسی_", "مهندسی-"]
    
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
            if name.startswith("_") or name.startswith("-"):
                 name = name[1:].strip()
            break
    
    if name == "مواد":
        name = "مواد_و_متالورژی"
            
    return name

def clean_course_name(course_name):
    cleaned = course_name.strip()
    # Remove space before digits: "فیزیک 1" -> "فیزیک1"
    cleaned = re.sub(r'\s+(\d+)', r'\1', cleaned)
    return cleaned

def split_courses(course_str):
    if not course_str:
        return []
    
    normalized = normalize_text(course_str)
    #  "2 course names combined", often with -.
    parts = re.split(r'[-–—,،|/\\+]+', normalized)
    
    courses = []
    for part in parts:
        cleaned = clean_course_name(part)
        if cleaned and len(cleaned) > 1: # Filter out single chars or empty strings
            courses.append(cleaned)
    return courses

def main():
    input_file = os.path.join("data", "processed", "processed_comments.json")
    output_file = os.path.join("data", "processed", "normalized_comments.json")
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return
        
    normalized_data = []
    
    for entry in data:
        new_entry = entry.copy()
        
        if "professor_name" in new_entry:
            new_entry["professor_name"] = normalize_professor_name(new_entry["professor_name"])
            
        if "faculty" in new_entry:
            new_entry["faculty"] = normalize_faculty_name(new_entry["faculty"])
     
        if "courses" in new_entry:
            # First normalize the string for other uses if needed, but we mainly want the list
            course_str = new_entry["courses"]
            new_entry["course_list"] = split_courses(course_str)
            # We keep the original (normalized) string too? Or just replace it?
            new_entry["courses"] = split_courses(course_str)
            
        normalized_data.append(new_entry)
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully normalized {len(normalized_data)} entries.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
