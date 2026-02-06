import json
import re
import os

def extract_text(text_obj):
    if isinstance(text_obj, str):
        return text_obj
    if isinstance(text_obj, list):
        return "".join([t["text"] if isinstance(t, dict) else t for t in text_obj])
    return ""

def parse_message(full_text):
    data = {}
    lines = full_text.split('\n')
    
    # Extract Professor Name, Faculty, and Courses from the top lines
    for line in lines[:5]: # Usually in the first few lines
        if "🧑‍🏫" in line:
            data["professor_name"] = line.replace("🧑‍🏫", "").strip()
        elif "🏫" in line and "🧑‍🏫" not in line: # Check for school emoji but not the teacher emoji
            faculty = line.replace("🏫", "").strip()
            data["faculty"] = faculty.replace("#", "").strip()
        elif "📒" in line:
            data["courses"] = line.replace("📒", "").strip()
        
   
    def extract_field(key, text):
        # Look for the key, then skip any whitespace/delimiters like ┘ or ┤
        pattern = re.escape(key) + r"\n\s*[┘┤]\s*(.*?)(?:\n\n|\n[^\s┘┤]|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    data["teaching_resources"] = extract_field("منابع آموزش", full_text)
    data["attendance"] = extract_field("حضور و غیاب", full_text)
    data["exam_adequacy"] = extract_field("منابع معرفی شده برای امتحان کافی است؟", full_text)
    data["grading"] = extract_field("وضعیت نمره دادن:", full_text)
    data["contact"] = extract_field("راه ارتباطی:", full_text)
    data["term"] = extract_field("ترمی که دانشجو با این استاد کلاس داشته:", full_text)
    
    # Extract Student Evaluation (Grades)
    eval_section = re.search(r"ارزیابی دانشجو:\n(.*?)(?:\n\n|\n[^\s┤┘]|$)", full_text, re.DOTALL)
    if eval_section:
        eval_text = eval_section.group(1)
        eval_data = {}
        for line in eval_text.split('\n'):
            line_match = re.search(r"[┤┘]\s*(.*?):\s*(\d+)", line)
            if line_match:
                key = line_match.group(1).strip()
                val = int(line_match.group(2))
                eval_data[key] = val
        if eval_data:
            data["evaluation_scores"] = eval_data
        
    # Extract Description
    desc_match = re.search(r"توضیحات:\n\s*[┘┤]\s*(.*?)(?:\n~~~~~~~~~~~~~~~~~|\n\n|$)", full_text, re.DOTALL)
    if desc_match:
        data["description"] = desc_match.group(1).strip()
    elif "توضیحات:" in full_text:
        desc_pattern = r"توضیحات:\n\s*[┘┤]\s*(.*?)(?=\n~~~~~~~~~~~~~~~~~|$)"
        desc_match = re.search(desc_pattern, full_text, re.DOTALL)
        if desc_match:
             data["description"] = desc_match.group(1).strip()

    if not data or len(data) <= 3: 
        return None
        
    data["raw_text"] = full_text
    return data

def main():
    try:
        input_path = os.path.join("data", "raw", "result.json")
        with open(input_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return
    except json.JSONDecodeError:
        print("Error: Failed to decode result.json.")
        return
        
    processed_messages = []
    messages = json_data.get("messages", [])
    
    for msg in messages:
        if msg.get("type") == "message":
            full_text = extract_text(msg.get("text", ""))
            if "🧑‍🏫" in full_text:
                parsed = parse_message(full_text)
                if parsed:
                    # Add some metadata
                    parsed["message_id"] = msg.get("id")
                    parsed["date"] = msg.get("date")
                    processed_messages.append(parsed)
                    
    output_path = os.path.join("data", "processed", "processed_comments.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_messages, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully processed {len(processed_messages)} reviews.")
    print("Output saved to processed_comments.json")

if __name__ == "__main__":
    main()
