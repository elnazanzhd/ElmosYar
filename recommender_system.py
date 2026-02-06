import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os

class ProfessorRecommender:
    def __init__(self, data_path):
        self.data_path = data_path
        self.professors = {} # Map: professor_name -> profile
        self.course_map = defaultdict(set) # Map: course_name -> set(professor_names)
        self.df = None
        
    def load_and_process(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found.")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
         
        processed_rows = []
        
        for entry in raw_data:
            prof_name = entry.get('professor_name')
            if not prof_name:
                continue
                
            # Map Sentiment to Numeric
            bert_sentiment = entry.get('bert_sentiment')
            bert_score = entry.get('bert_score', 0.0)
            if bert_sentiment == 'POSITIVE':
                sentiment_val = bert_score
            elif bert_sentiment == 'NEGATIVE':
                sentiment_val = -bert_score
            else:
                sentiment_val = 0.0
                
            # Evaluation Score (Average)
            eval_scores = entry.get('evaluation_scores', {})
            if eval_scores:
                avg_score = sum(eval_scores.values()) / len(eval_scores)
            else:
                avg_score = None # Treat as missing
            
            row = {
                'professor_name': prof_name,
                'sentiment_val': sentiment_val,
                'avg_eval_score': avg_score,
                'grading': entry.get('feature_grading', 'unknown'),
                'attendance': entry.get('feature_attendance', 'unknown'),
                'resources': entry.get('feature_resources', 'unknown'),
                'exam_adequacy': entry.get('feature_exam_adequacy', 'unknown'),
                'faculty': entry.get('faculty', 'Unknown')
            }
            processed_rows.append(row)
            
            # Update Course Map
            courses = entry.get('course_list', [])
            if courses:
                for course in courses:
                    cleaned_course = course.strip()
                    if cleaned_course:
                        self.course_map[cleaned_course].add(prof_name)
        
        self.df = pd.DataFrame(processed_rows)
        
  
        self._build_profiles()
        
    def _build_profiles(self):
        grouped = self.df.groupby('professor_name')
        
        for name, group in grouped:
            profile = {
                'name': name,
                'review_count': len(group),
                'faculty': group['faculty'].iloc[0], # Just take the first one
                'avg_sentiment': group['sentiment_val'].mean(),
                'avg_eval_score': group['avg_eval_score'].mean(),
                
                # Calculate distribution of categorical features
                'grading_dist': group['grading'].value_counts(normalize=True).to_dict(),
                'attendance_dist': group['attendance'].value_counts(normalize=True).to_dict(),
                'resources_dist': group['resources'].value_counts(normalize=True).to_dict(),
                'exam_dist': group['exam_adequacy'].value_counts(normalize=True).to_dict()
            }
            
            # Determine Dominant Traits (if > 30% frequency)
            def get_dominant(dist, ignore='unknown'):
                filtered = {k: v for k, v in dist.items() if k != ignore}
                if not filtered:
                    return 'unknown'
                top_trait = max(filtered, key=filtered.get)
                if filtered[top_trait] > 0.3: # Threshold
                    return top_trait
                return 'mixed'
                
            profile['grading_style'] = get_dominant(profile['grading_dist'])
            profile['attendance_style'] = get_dominant(profile['attendance_dist'])
            
            self.professors[name] = profile
            
    def recommend(self, course_name=None, priorities=None):
        """
        priorities: dict with keys:
            - grading: 'lenient', 'fair', 'strict'
            - attendance: 'optional', 'strict', 'bonus'
            - min_score: float (0-10)
            - sort_by: 'sentiment', 'score', 'match'
        """
        if priorities is None:
            priorities = {}
            
        candidates = []
        
        # 1. Filter by Course
        if course_name:
            relevant_profs = set()
            for stored_course, profs in self.course_map.items():
                if course_name in stored_course: # e.g. "math" in "engineering math"
                    relevant_profs.update(profs)
            
            if not relevant_profs:
                return [], f"No professors found for course containing '{course_name}'"
        else:
            relevant_profs = self.professors.keys() # All professors
            
        # 2. Score Candidates
        results = []
        for prof_name in relevant_profs:
            if prof_name not in self.professors:
                continue
                
            profile = self.professors[prof_name]
            
         
            score = profile['avg_eval_score'] if not pd.isna(profile['avg_eval_score']) else 0
            if 'min_score' in priorities and score < priorities['min_score']:
                continue
                
            # Calculate Match Score
            match_score = 0.0
            max_match_score = 0.0
            
            # Grading Preference
            if 'grading' in priorities:
                target = priorities['grading']
                # Add score based on probability of this trait
                prob = profile['grading_dist'].get(target, 0.0)
                match_score += prob * 2 # Weight = 2
                max_match_score += 2
                
            # Attendance Preference
            if 'attendance' in priorities:
                target = priorities['attendance']
                prob = profile['attendance_dist'].get(target, 0.0)
                match_score += prob * 1 # Weight = 1
                max_match_score += 1
                
            # Normalize Match Score (0 to 1)
            final_match = 0
            if max_match_score > 0:
                final_match = match_score / max_match_score
            
            # Combined Score: 
            # 50% Quality (Sentiment + Eval) + 50% Preference Match
    
            norm_sentiment = (profile['avg_sentiment'] + 1) / 2
            
            # Normalize Eval (0 to 10) -> (0 to 1)
            norm_eval = score / 10.0
            
            quality_score = (norm_sentiment + norm_eval) / 2
            
            # If no preferences given, rank by quality only
            if max_match_score == 0:
                final_score = quality_score
            else:
                final_score = (quality_score * 0.4) + (final_match * 0.6) # Bias towards preferences
                
            results.append({
                'name': profile['name'],
                'faculty': profile['faculty'],
                'score': score,
                'sentiment': profile['avg_sentiment'],
                'grading': profile['grading_style'],
                'attendance': profile['attendance_style'],
                'match_prob': final_match,
                'final_score': final_score,
                'review_count': profile['review_count']
            })
            
        # 3. Sort
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results, "Success"

def main():
    rec = ProfessorRecommender(os.path.join('data', 'normalized_comments.json'))
    print("Loading data...")
    rec.load_and_process()
    print("System Ready.\n")
    
    while True:
        print("-" * 50)
        print("🤖 Professor Recommender System")
        print("-" * 50)
        
        course = input("Enter Course Name (or 'exit'): ").strip()
        if course.lower() == 'exit':
            break
            
        print("\nPreferences (Press Enter to skip):")
        grading = input("Grading (lenient/fair/strict): ").strip()
        attendance = input("Attendance (optional/bonus/strict): ").strip()
        
        priorities = {}
        if grading: priorities['grading'] = grading
        if attendance: priorities['attendance'] = attendance
        
        print("\nSearching...")
        results, msg = rec.recommend(course, priorities)
        
        if not results:
            print(f"❌ {msg}")
        else:
            print(f"✅ Found {len(results)} professors.\n")
            print(f"{'RANK':<4} {'NAME':<20} {'FACULTY':<15} {'SCORE':<6} {'GRADING':<10} {'ATTENDANCE':<10} {'MATCH':<6}")
            print("-" * 80)
            
            for i, r in enumerate(results[:10]): # Show top 10
                match_str = f"{int(r['match_prob']*100)}%" if priorities else "N/A"
                print(f"{i+1:<4} {r['name']:<20} {r['faculty']:<15} {r['score']:<6.1f} {r['grading']:<10} {r['attendance']:<10} {match_str:<6}")
                
        print("\n")

if __name__ == "__main__":
    main()
