import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
from datetime import datetime
import hashlib
import os
from pathlib import Path
import logging
import base64
from typing import Dict, List, Any, Optional
import mimetypes
import threading
import pymongo
from pymongo import MongoClient

# --- Optional .env loader (no external dependency) ---
def _load_env_from_dotenv():
    """Load environment variables from a local .env file if present.
    Only sets variables that are not already present in os.environ."""
    try:
        env_path = Path('.env')
        if env_path.exists():
            with env_path.open('r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            try:
                logger.info("Loaded environment variables from .env")
            except Exception:
                # logger may not be configured yet; ignore
                pass
    except Exception as e:
        try:
            logger.warning(f"Failed to load .env: {e}")
        except Exception:
            pass

def _apply_fix_to_question_in_place(question: Dict, fixed: Dict) -> None:
    """Mutate a question dict in-place with a sanitized fix object."""
    question['text'] = (
        (fixed.get('added_context') + ' ') if fixed.get('added_context') else ''
    ) + fixed.get('text', question.get('text', ''))
    question['options'] = fixed['options']
    question['answer'] = fixed['answer']

def _prewarm_sanitizer_all_questions() -> Dict[str, int]:
    """Disabled - question fixing functionality removed.
    Returns empty stats dict."""
    return {'total': 0, 'fixed': 0, 'skipped': 0}

def _rule_based_fix_question(question: Dict) -> Optional[Dict]:
    # Ratio + sum: e.g., boys:girls = a:b, boys+girls=N, find boys and girls
    import re as _re2
    m_ratio_sum = _re2.search(r"boys\s*[:=]\s*girls\s*[=:]?\s*(\d+)\s*[:/]\s*(\d+)[^\d]+boys\s*\+\s*girls\s*=\s*(\d+)", low)
    if m_ratio_sum:
        a = int(m_ratio_sum.group(1))
        b = int(m_ratio_sum.group(2))
        total = int(m_ratio_sum.group(3))
        s = a + b
        if total % s == 0:
            k = total // s
            boys = a * k
            girls = b * k
            correct_pair = f"Boys = {boys}, Girls = {girls}"
            # Generate plausible distractors
            distractors = []
            for delta in [-20, -10, 10, 20, -5, 5]:
                b2 = boys + delta
                g2 = girls - delta
                if b2 > 0 and g2 > 0 and (b2 != boys or g2 != girls):
                    distractors.append(f"Boys = {b2}, Girls = {g2}")
                if len(distractors) >= 3:
                    break
            values = [correct_pair] + distractors[:3]
            # Shuffle deterministically
            import random as _rnd2
            _rnd2.seed(str(question.get('id')))
            _rnd2.shuffle(values)
            opts = {lab: values[i] for i, lab in enumerate(['A','B','C','D'])}
            answer = [lab for lab, val in opts.items() if val == correct_pair][0]
            return {
                'text': text,
                'added_context': '',
                'options': opts,
                'answer': answer,
                'notes': 'rule-based ratio+sum pair fix applied'
            }
        else:
            # Not divisible, so not possible
            opts = {'A': 'Not possible with given data', 'B': 'Cannot be determined', 'C': 'Insufficient information', 'D': 'None of these'}
            return {
                'text': text,
                'added_context': '',
                'options': opts,
                'answer': 'A',
                'notes': 'rule-based ratio+sum not possible'
            }
    # General context enrichment by topic
    topic = (question.get('topic') or '').strip().lower()
    text = (question.get('text') or question.get('question_text') or '').strip()
    low = text.lower()
    context_map = {
        'probability': "Suppose two people, A and B, are participating in a race.",
        'varc': "Read the following passage and answer the question:",
        'verbal': "Read the following passage and answer the question:",
        'logic': "Consider the following logical scenario:",
        'reasoning': "Consider the following logical scenario:",
        'geometry': "Refer to the following geometric figure or description:",
        'algebra': "Solve the following algebraic problem:",
        'arithmetic': "Solve the following arithmetic problem:",
        'modern math': "Solve the following modern math problem:",
        'number system': "Solve the following number system problem:",
        'mensuration': "Refer to the following mensuration scenario:",
        'data interpretation': "Interpret the following data and answer the question:",
        'statistics': "Analyze the following statistical data:",
        'trigonometry': "Solve the following trigonometry problem:",
        'coordinate geometry': "Refer to the following coordinate geometry scenario:",
        'sets': "Consider the following set theory scenario:",
        'functions': "Analyze the following function:",
        'series': "Analyze the following series:",
        'permutation': "Consider the following permutation and combination scenario:",
        'combination': "Consider the following permutation and combination scenario:",
        'time and work': "Solve the following time and work problem:",
        'time and distance': "Solve the following time and distance problem:",
        'simple interest': "Solve the following simple interest problem:",
        'compound interest': "Solve the following compound interest problem:",
        'profit and loss': "Solve the following profit and loss problem:",
        'ratio': "Solve the following ratio and proportion problem:",
        'proportion': "Solve the following ratio and proportion problem:",
        'mixtures': "Solve the following mixtures and alligation problem:",
        'alligation': "Solve the following mixtures and alligation problem:",
        'partnership': "Solve the following partnership problem:",
        'average': "Solve the following average problem:",
        'age': "Solve the following age problem:",
        'calendar': "Solve the following calendar problem:",
        'clock': "Solve the following clock problem:",
        'direction': "Refer to the following direction scenario:",
        'blood relation': "Consider the following family tree or relationship:",
        'coding': "Decode the following code or pattern:",
        'decoding': "Decode the following code or pattern:",
        'puzzle': "Solve the following logical puzzle:",
        'sitting arrangement': "Consider the following seating arrangement:",
        'input output': "Analyze the following input-output pattern:",
        'syllogism': "Analyze the following syllogism:",
        'statement conclusion': "Analyze the following statement and conclusion:",
        'statement assumption': "Analyze the following statement and assumption:",
        'statement argument': "Analyze the following statement and argument:",
        'statement course of action': "Analyze the following statement and course of action:",
        'direction sense': "Refer to the following direction sense scenario:",
        'seating arrangement': "Consider the following seating arrangement:",
        'logical reasoning': "Consider the following logical reasoning scenario:",
    }
    # Only add context if not already present and not already handled above
    # Special handling for VARC/verbal: if prompt says 'Read the following passage' but no passage is present, inject a generic sample passage
    if topic in ['varc', 'verbal', 'reading'] and 'read the following passage' in low:
        # Heuristic: if text does not contain a quoted or multi-line passage, add a generic one
        if len(text.splitlines()) < 2 and 'passage:' not in low:
            sample_passage = (
                "Passage: Mobile phones have become an essential part of modern life. "
                "Many social media influencers prefer phones with high storage capacity to store photos and videos. "
                "Some brands are more popular among influencers due to their advanced features."
            )
            return {
                'text': text,
                'added_context': sample_passage,
                'options': question.get('options', {'A':'','B':'','C':'','D':''}),
                'answer': question.get('answer','A'),
                'notes': 'rule-based sample passage injected for VARC'
            }
    if topic in context_map and not any(context_map[topic].lower() in low for _ in [0]):
        return {
            'text': text,
            'added_context': context_map[topic],
            'options': question.get('options', {'A':'','B':'','C':'','D':''}),
            'answer': question.get('answer','A'),
            'notes': f'rule-based context added for topic: {topic}'
        }
    """Deterministic fixes for common math patterns when Gemini is unavailable.
    Returns a dict similar to _gemini_fix_question or None if no rule applies."""
    import re as _re
    text = (question.get('text') or question.get('question_text') or '').strip()
    low = text.lower()

    # Pattern: ratio of boys to girls is a:b. If boys is N, what is girls? (or vice versa)
    m_ratio = _re.search(r"ratio\s+of\s+boys\s*to\s*girls\s*is\s*(\d+)\s*[:/]\s*(\d+)", low)
    if m_ratio:
        a = int(m_ratio.group(1))
        b = int(m_ratio.group(2))
        m_boys = _re.search(r"if\s+boys\s+(?:is|are)\s*(\d+)", low)
        m_girls = _re.search(r"if\s+girls\s+(?:is|are)\s*(\d+)", low)
        if m_boys or m_girls:
            if m_boys:
                n = int(m_boys.group(1))
                # boys : girls = a : b => girls = n * b / a
                if a != 0 and (n * b) % a == 0:
                    girls = (n * b) // a
                    correct = girls
                else:
                    # Inconsistent data, include Not possible option
                    correct = None
            else:
                n = int(m_girls.group(1))
                # boys : girls = a : b => boys = n * a / b
                if b != 0 and (n * a) % b == 0:
                    boys = (n * a) // b
                    correct = boys
                else:
                    correct = None

            # Build options
            opts = {}
            labels = ['A','B','C','D']
            if correct is not None:
                # plausible distractors near the correct value
                distractors = []
                for delta in (-50, -25, 25, 50, -10, 10, -5, 5):
                    cand = correct + delta
                    if cand > 0 and cand != correct:
                        distractors.append(cand)
                    if len(distractors) >= 3:
                        break
                values = [correct] + distractors[:3]
                # Shuffle deterministically based on id for consistency
                import random as _rnd
                _rnd.seed(str(question.get('id')))
                _rnd.shuffle(values)
                for i, lab in enumerate(labels):
                    opts[lab] = str(values[i])
                answer = labels[values.index(correct)]
            else:
                # Not solvable cleanly
                opts = {'A': 'Not possible with given data', 'B': 'Cannot be determined', 'C': 'Insufficient information', 'D': 'None of these'}
                answer = 'A'

            return {
                'text': text,
                'added_context': '',
                'options': opts,
                'answer': answer,
                'notes': 'rule-based ratio fix applied'
            }

        return None
    # Probability context enrichment
    text = (question.get('text') or question.get('question_text') or '').strip()
    low = text.lower()
    # If the question is about probability and lacks a scenario, add a generic context
    if 'probability' in low and ('neither' in low or 'both' in low or 'at least' in low or 'at most' in low or 'only one' in low):
        # Only add context if not already present
        if not any(word in low for word in ['bag', 'dice', 'coin', 'deck', 'cards', 'urn', 'marbles', 'balls', 'students', 'people', 'race', 'event', 'experiment']):
            added_context = "Suppose two people, A and B, are participating in a race. "
            return {
                'text': text,
                'added_context': added_context,
                'options': question.get('options', {'A':'','B':'','C':'','D':''}),
                'answer': question.get('answer','A'),
                'notes': 'rule-based probability context added'
            }

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load environment variables
_load_env_from_dotenv()

# MongoDB setup
try:
    MONGODB_URI = os.environ.get('MONGODB_URI')
    if not MONGODB_URI:
        raise Exception("MONGODB_URI not found in environment variables")
    
    client = MongoClient(MONGODB_URI)
    db = client.adaptiq_db
    students_collection = db.students
    
    # Test the connection
    client.admin.command('ping')
    print("✅ MongoDB connection successful!")
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    # Fallback to file-based storage
    client = None
    db = None
    students_collection = None

# Global variables
trained_model = None
student_sessions = {}
image_cache = {}

# --- Student history persistence helpers ---
HISTORY_DIR = Path('data') / 'student_history'
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def _compute_profile_id(name: str, grade: str) -> str:
    base = f"{(name or '').strip().lower()}|{(grade or '').strip().lower()}"
    return hashlib.sha1(base.encode('utf-8')).hexdigest()[:16]

def _load_history(profile_id: str) -> dict:
    """Load student history from MongoDB or fallback to file storage"""
    if students_collection is not None:
        try:
            # Try to load from MongoDB
            doc = students_collection.find_one({"profile_id": profile_id})
            if doc:
                # Remove MongoDB's _id field and return the data
                doc.pop('_id', None)
                return doc
            else:
                # Return default structure if no document found
                return {"profile": {}, "sessions": {}, "responses": []}
        except Exception as e:
            print(f"MongoDB load error: {e}, falling back to file storage")
    
    # Fallback to file-based storage
    fp = HISTORY_DIR / f"{profile_id}.json"
    if not fp.exists():
        return {"profile": {}, "sessions": {}, "responses": []}
    try:
        with fp.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"profile": {}, "sessions": {}, "responses": []}

def _save_history(profile_id: str, data: dict):
    """Save student history to MongoDB and optionally keep file backup"""
    if students_collection is not None:
        try:
            # Ensure profile_id is in the data
            data['profile_id'] = profile_id
            data['last_updated'] = datetime.now()
            
            # Use upsert to update or insert the document
            students_collection.replace_one(
                {"profile_id": profile_id}, 
                data, 
                upsert=True
            )
            print(f"✅ Student data saved to MongoDB for profile: {profile_id}")
            return
        except Exception as e:
            print(f"MongoDB save error: {e}, falling back to file storage")
    
    # Fallback to file-based storage
    fp = HISTORY_DIR / f"{profile_id}.json"
    try:
        with fp.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"File save error: {e}")

def get_all_students():
    """Get all students from MongoDB"""
    if students_collection is not None:
        try:
            students = list(students_collection.find({}, {"_id": 0}))
            return students
        except Exception as e:
            print(f"Error fetching all students: {e}")
    return []

def get_student_stats():
    """Get overall student statistics"""
    if students_collection is not None:
        try:
            total_students = students_collection.count_documents({})
            active_students = students_collection.count_documents({
                "last_updated": {"$gte": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)}
            })
            
            # Aggregate response statistics
            pipeline = [
                {"$unwind": "$responses"},
                {"$group": {
                    "_id": None,
                    "total_responses": {"$sum": 1},
                    "correct_responses": {"$sum": {"$cond": ["$responses.is_correct", 1, 0]}}
                }}
            ]
            
            result = list(students_collection.aggregate(pipeline))
            total_responses = result[0]["total_responses"] if result else 0
            correct_responses = result[0]["correct_responses"] if result else 0
            overall_accuracy = (correct_responses / total_responses * 100) if total_responses > 0 else 0
            
            return {
                "total_students": total_students,
                "active_students_today": active_students,
                "total_responses": total_responses,
                "overall_accuracy": round(overall_accuracy, 2)
            }
        except Exception as e:
            print(f"Error getting student stats: {e}")
    
    return {
        "total_students": 0,
        "active_students_today": 0,
        "total_responses": 0,
        "overall_accuracy": 0
    }

def delete_student_data(profile_id: str):
    """Delete a student's data from MongoDB"""
    if students_collection is not None:
        try:
            result = students_collection.delete_one({"profile_id": profile_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting student data: {e}")
    return False

class AdaptiveAssessmentEngine:
    """
    Enhanced adaptive assessment engine with image support
    """
    
    def __init__(self, model_data: Dict):
        self.model_data = model_data
        self.questions = {q['id']: q for q in model_data['questions']}
        self.item_parameters = model_data['question_parameters']
        self.topics = model_data['topics']
        self.image_mappings = model_data.get('image_mappings', {})
        
        # IRT Model parameters - Start with Very Easy questions
        self.default_ability = -1.5  # Starting ability level (ensures Very Easy start)
        self.ability_range = (-3.0, 3.0)  # Ability range
        
        logger.info(f"Initialized adaptive engine with {len(self.questions)} questions")
        logger.info(f"Questions with images: {sum(1 for q in self.questions.values() if q.get('has_image'))}")
    
    def calculate_probability(self, ability: float, item_id: str) -> float:
        """Calculate probability of correct response using 2PL IRT model"""
        if item_id not in self.item_parameters:
            return 0.5  # Default probability
        
        params = self.item_parameters[item_id]
        a = params['discrimination']  # Discrimination parameter
        b = params['difficulty']      # Difficulty parameter
        c = params.get('guessing', 0.25)  # Guessing parameter (default 25% for 4-option MCQ)
        
        try:
            # 3PL model: P(correct) = c + (1-c) * (1 / (1 + exp(-a(θ-b))))
            prob = c + (1 - c) * (1 / (1 + np.exp(-a * (ability - b))))
            return max(0.01, min(0.99, prob))  # Clamp probability
        except (OverflowError, ZeroDivisionError):
            return 0.5
    
    def select_next_question(self, student_id: str, answered_questions: List[str] = None, 
                           target_topic: str = None, target_difficulty: str = None) -> Optional[Dict]:
        """Select the next most informative question for the student using advanced adaptive algorithm"""
        
        if answered_questions is None:
            answered_questions = []

        # Get student's current ability estimate and performance history
        ability = self.get_student_ability(student_id)
        session = student_sessions.get(student_id, {})
        responses = session.get('responses', [])

        # Exclude all previously answered questions for this profile (across all sessions)
        profile_id = session.get('profile_id')
        prev_answered = set()
        if profile_id:
            hist = _load_history(profile_id)
            prev_answered = set(r['question_id'] for r in hist.get('responses', []) if r.get('question_id'))

        logger.info(f"Selecting question for student {student_id} with ability {ability:.3f}")
        logger.info(f"Already answered {len(answered_questions)} questions: {answered_questions}")

        # Get available questions (ensure we exclude all previously answered questions)
        all_answered = set(answered_questions) | prev_answered
        available_questions = [
            q for q in self.questions.values()
            if (q['id'] not in all_answered and
                str(q['id']) not in [str(qid) for qid in all_answered] and
                self.is_question_complete(q))
        ]
        
        logger.info(f"Total available questions: {len(available_questions)} out of {len(self.questions)}")
        
        # Filter by topic if specified
        if target_topic:
            available_questions = [
                q for q in available_questions 
                if q['topic'].lower() == target_topic.lower()
            ]
        
        # Adaptive difficulty targeting based on student ability
        if not target_difficulty:
            # For first question, always start with Very Easy
            if len(responses) == 0:
                target_difficulty = "Very Easy"
                logger.info(f"First question - starting with Very Easy for new student {student_id}")
            else:
                target_difficulty = self.get_optimal_difficulty_for_ability(ability)
                logger.info(f"Auto-selected difficulty: {target_difficulty} for ability {ability:.3f}")
        
        # Filter by difficulty with some flexibility
        if target_difficulty:
            primary_questions = [
                q for q in available_questions 
                if q['difficulty'].lower() == target_difficulty.lower()
            ]
            
            # If not enough questions at target difficulty, expand to adjacent difficulties
            if len(primary_questions) < 10:
                adjacent_difficulties = self.get_adjacent_difficulties(target_difficulty)
                secondary_questions = [
                    q for q in available_questions 
                    if q['difficulty'] in adjacent_difficulties
                ]
                available_questions = primary_questions + secondary_questions
            else:
                available_questions = primary_questions
        
        if not available_questions:
            logger.warning(f"No available questions for student {student_id}")
            return None
        
        # Enhanced question selection algorithm
        best_question = self.select_optimal_question(available_questions, ability, responses)
        
        # Prepare question for serving (include image data if needed)
        if best_question:
            best_question = self.prepare_question_for_serving(best_question)
            logger.info(f"Selected question {best_question['id']} (difficulty: {best_question['difficulty']}, topic: {best_question['topic']})")
        
        return best_question
    
    def get_optimal_difficulty_for_ability(self, ability: float) -> str:
        """Map student ability to optimal question difficulty with step-by-step progression"""
        # More gradual progression - always start Very Easy and move up slowly
        if ability <= -1.0:
            return "Very Easy"
        elif ability <= 0.0:
            return "Easy" 
        elif ability <= 1.0:
            return "Moderate"
        else:
            return "Difficult"
    
    def get_adjacent_difficulties(self, target_difficulty: str) -> List[str]:
        """Get adjacent difficulty levels for flexibility"""
        difficulty_order = ["Very Easy", "Easy", "Moderate", "Difficult"]
        
        try:
            index = difficulty_order.index(target_difficulty)
            adjacent = []
            if index > 0:
                adjacent.append(difficulty_order[index - 1])
            if index < len(difficulty_order) - 1:
                adjacent.append(difficulty_order[index + 1])
            return adjacent
        except ValueError:
            return ["Easy", "Moderate"]
    
    def select_optimal_question(self, available_questions: List[Dict], ability: float, responses: List[Dict]) -> Optional[Dict]:
        """Advanced question selection using multiple criteria"""
        
        if not available_questions:
            return None
        
        # Get recently used topics and difficulties for diversity
        recent_topics = [self.questions[r['question_id']]['topic'] for r in responses[-5:] if r['question_id'] in self.questions]
        recent_difficulties = [self.questions[r['question_id']]['difficulty'] for r in responses[-3:] if r['question_id'] in self.questions]
        
        question_scores = []
        
        for question in available_questions:
            score = 0
            
            # 1. Information value (Fisher Information)
            information = self.calculate_information(ability, question['id'])
            score += information * 0.35  # 35% weight
            
            # 2. Ability-difficulty match
            difficulty_match = self.calculate_difficulty_match(ability, question)
            score += difficulty_match * 0.25  # 25% weight
            
            # 3. Enhanced topic diversity (stronger penalty for recently used topics)
            topic_diversity = self.calculate_enhanced_topic_diversity(question, recent_topics, recent_difficulties)
            score += topic_diversity * 0.25  # 25% weight (increased)
            
            # 4. Question quality indicators
            quality_score = self.calculate_question_quality_score(question)
            score += quality_score * 0.1  # 10% weight
            
            # 5. Add small randomness to prevent always picking same "optimal" question
            randomness = np.random.uniform(0.0, 0.05)
            score += randomness
            
            question_scores.append((question, score))
        
        # Sort by score and return best question with some randomness in top candidates
        question_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top 3 candidates to add variety
        top_candidates = question_scores[:min(3, len(question_scores))]
        selected = np.random.choice([q[0] for q in top_candidates])
        
        # Log selected question details for debugging
        logger.info(f"Selected question {selected['id']} (topic: {selected['topic']}, difficulty: {selected['difficulty']})")
        logger.info(f"Question text preview: '{selected['text'][:50]}...'")
        logger.info(f"Options available: {list(selected.get('options', {}).keys())}")
        
        return selected
    
    def calculate_difficulty_match(self, ability: float, question: Dict) -> float:
        """Calculate how well question difficulty matches student ability"""
        difficulty = question['difficulty']
        
        # Map difficulty to numeric scale
        difficulty_map = {
            "Very Easy": -1.5,
            "Easy": -0.5,
            "Moderate": 0.5,
            "Difficult": 1.5
        }
        
        difficulty_value = difficulty_map.get(difficulty, 0)
        
        # Calculate match score (inverse of distance)
        distance = abs(ability - difficulty_value)
        match_score = max(0, 1 - distance / 3)  # Normalize to 0-1
        
        return match_score
    
    def calculate_topic_diversity(self, question: Dict, responses: List[Dict]) -> float:
        """Encourage topic diversity in question selection"""
        if not responses:
            return 1.0
        
        question_topic = question['topic']
        recent_topics = [
            self.questions[r['question_id']]['topic'] 
            for r in responses[-5:] 
            if r['question_id'] in self.questions
        ]
        
        # Count occurrences of this topic in recent questions
        topic_count = recent_topics.count(question_topic)
        
        # Return diversity score (lower if topic appears frequently)
        return max(0.1, 1 - (topic_count * 0.2))
    
    def calculate_enhanced_topic_diversity(self, question: Dict, recent_topics: List[str], recent_difficulties: List[str]) -> float:
        """Enhanced topic and difficulty diversity calculation"""
        diversity_score = 1.0
        
        question_topic = question['topic']
        question_difficulty = question['difficulty']
        
        # Penalty for repeating topics (stronger penalty for more recent repetitions)
        for i, topic in enumerate(reversed(recent_topics)):
            if topic == question_topic:
                # More recent = higher penalty (exponential decay)
                penalty = 0.4 * (0.7 ** i)  # Penalty decreases with distance
                diversity_score -= penalty
        
        # Additional penalty for repeating same difficulty too often
        difficulty_count = recent_difficulties.count(question_difficulty)
        if difficulty_count > 1:
            diversity_score -= 0.2 * (difficulty_count - 1)
        
        # Bonus for questions from underrepresented topics
        unique_recent_topics = set(recent_topics)
        if question_topic not in unique_recent_topics and len(unique_recent_topics) > 0:
            diversity_score += 0.3  # Bonus for new topic
        
        return max(0.1, diversity_score)
    
    def is_question_complete(self, question: Dict) -> bool:
        """Check if a question has complete and valid content"""
        # Check if question text exists and is meaningful
        text = question.get('text', '').strip()
        if len(text) < 20:  # Very short questions are likely incomplete
            return False
        
        # Ensure question ends with proper punctuation or question mark
        if not (text.endswith('?') or text.endswith('.') or text.endswith(':') or '?' in text):
            return False
        
        # Check if options exist and are complete
        options = question.get('options', {})
        if not options or len(options) < 4:
            return False
        
        # Check if all options have content
        for opt_key in ['A', 'B', 'C', 'D']:
            option_text = options.get(opt_key, '').strip()
            if not option_text or len(option_text) < 2:  # Options should have meaningful content
                return False
        
        # Check if answer exists
        answer = question.get('answer', '').strip()
        if not answer or answer not in ['A', 'B', 'C', 'D']:
            return False
        
        text_lower = text.lower()
        
        # Filter out questions that reference missing content
        incomplete_reference_patterns = [
            'pick the option that gives the correct order of the following sentences',
            'arrange the following sentences',
            'the correct order of the following sentences',
            'arrange the following words',
            'arrange the following in order',
            'complete the following series:',
            'the following sentences in correct order',
            'order the following',
            'sequence the following',
            'decide which of the conclusions logically follows',
            'which of the following can be inferred',
            'complete the following passage',
            'fill in the blanks in the following',
            'referring to the above passage',
            'based on the above information',
            'from the given information',
            'the above figure shows',
            'in the figure above',
            'as shown in the diagram',
            'choose the most logical order of sentences',
            'choose the correct order of sentences',
            'arrange these sentences in logical order',
            'logical order of sentences',
            'construct a coherent paragraph',
            'most logical sequence',
            'proper sequence of sentences',
        ]
        
        # Check for incomplete reference patterns
        for pattern in incomplete_reference_patterns:
            if pattern in text_lower:
                # If it references "following" content but the question is short, it's likely incomplete
                if 'following' in pattern and len(text) < 150:
                    return False
                # If it references ordering/arranging but question is too short to contain actual content
                if any(word in pattern for word in ['order', 'arrange', 'sequence', 'logical']) and len(text) < 120:
                    return False
                # If it references diagrams/figures but has no image, it's incomplete
                if any(ref in pattern for ref in ['figure', 'diagram', 'above']) and not question.get('has_image'):
                    return False
                # If it asks to construct coherent paragraph but doesn't provide sentences
                if 'coherent paragraph' in pattern and len(text) < 100:
                    return False
        
        # Filter out questions that are just fragments or incomplete
        problematic_exact_patterns = [
            'design and development',
            'prototype creation', 
            'market survey',
            'manufacturing setup',
            'marketing and launch',
        ]
        
        # Direct pattern matches for exact phrases
        if text_lower in problematic_exact_patterns:
            return False
        
        # Additional checks for incomplete logical reasoning questions
        if ('decide which' in text_lower or 'conclusions' in text_lower) and len(text) < 50:
            return False
            
        # Check for questions that are just series completion without proper content
        if 'complete the following series:' in text_lower and len(text) < 40:
            return False
        
        # Check for questions that only have single letters or numbers as options (likely incomplete)
        option_values = [options.get(key, '').strip() for key in ['A', 'B', 'C', 'D']]
        if all(len(opt) <= 4 and opt.isalnum() for opt in option_values):
            # All options are very short alphanumeric - check if question provides context
            if not any(keyword in text_lower for keyword in ['arrange', 'order', 'sequence', 'pattern', 'series']):
                # If no ordering context is clear, this might be incomplete
                if len(text) < 60:  # Short question with short options is suspicious
                    return False
            
        # Filter out questions that seem to be just category labels
        category_patterns = ['arithmetic', 'algebra', 'geometry', 'reasoning', 'mathematics']
        if text_lower in category_patterns:
            return False
            
        return True
    
    def calculate_question_quality_score(self, question: Dict) -> float:
        """Calculate question quality based on various factors"""
        score = 0.5  # Base score
        
        # Bonus for questions with images (often more engaging)
        if question.get('has_image'):
            score += 0.3
        
        # Bonus for complete option sets
        options = question.get('options', {})
        if len(options) >= 4:
            score += 0.2
        
        # Bonus for reasonable question length
        text_length = len(question.get('text', ''))
        if 20 <= text_length <= 200:
            score += 0.1
        
        return min(1.0, score)
    
    def calculate_information(self, ability: float, item_id: str) -> float:
        """Calculate Fisher information for an item at given ability level"""
        prob = self.calculate_probability(ability, item_id)
        
        if item_id not in self.item_parameters:
            return 0.0
        
        a = self.item_parameters[item_id]['discrimination']
        c = self.item_parameters[item_id].get('guessing', 0.25)
        
        try:
            # Fisher information for 3PL model
            numerator = (a ** 2) * ((prob - c) ** 2) * (1 - prob)
            denominator = prob * ((1 - c) ** 2)
            
            if denominator > 0:
                return numerator / denominator
            else:
                return 0.0
        except (ZeroDivisionError, OverflowError):
            return 0.0
    
    def update_student_ability(self, student_id: str, question_id: str, response: bool):
        """Update student ability based on response using EAP estimation"""
        
        if student_id not in student_sessions:
            student_sessions[student_id] = {
                'ability': self.default_ability,
                'responses': [],
                'ability_history': [self.default_ability],
                'start_time': datetime.now(),
                'last_update': datetime.now()
            }
        
        session = student_sessions[student_id]
        
        # Add response to history
        session['responses'].append({
            'question_id': question_id,
            'response': response,
            'timestamp': datetime.now(),
            'probability': self.calculate_probability(session['ability'], question_id)
        })
        
        # Track answered questions to avoid repetition
        if 'answered_questions' not in session:
            session['answered_questions'] = []
        if question_id not in session['answered_questions']:
            session['answered_questions'].append(question_id)
        
        # Get current question difficulty to determine step size
        current_question = self.questions.get(question_id, {})
        current_difficulty = current_question.get('difficulty', 'Moderate')
        
        # Implement step-by-step ability progression
        current_ability = session['ability']
        
        if response:  # Correct answer
            # Move up one difficulty level gradually
            if current_difficulty == "Very Easy":
                new_ability = max(current_ability + 0.7, -0.5)  # Move toward Easy
            elif current_difficulty == "Easy":  
                new_ability = max(current_ability + 0.6, 0.2)   # Move toward Moderate
            elif current_difficulty == "Moderate":
                new_ability = max(current_ability + 0.5, 1.2)   # Move toward Difficult
            else:  # Difficult
                new_ability = current_ability + 0.3  # Continue improving
        else:  # Wrong answer
            # Move down one difficulty level gradually
            if current_difficulty == "Difficult":
                new_ability = min(current_ability - 0.5, 0.5)   # Move toward Moderate
            elif current_difficulty == "Moderate":
                new_ability = min(current_ability - 0.6, -0.2)  # Move toward Easy
            elif current_difficulty == "Easy":
                new_ability = min(current_ability - 0.7, -1.2)  # Move toward Very Easy
            else:  # Very Easy
                new_ability = current_ability - 0.3  # Further down in Very Easy
        
        # Ensure ability stays within bounds
        new_ability = max(self.ability_range[0], min(self.ability_range[1], new_ability))
        
        session['ability'] = new_ability
        session['ability_history'].append(new_ability)
        session['last_update'] = datetime.now()
        
        old_diff = self.get_optimal_difficulty_for_ability(current_ability)
        new_diff = self.get_optimal_difficulty_for_ability(new_ability)
        
        logger.info(f"Updated ability for {student_id}: {current_ability:.3f} -> {new_ability:.3f} "
                   f"({old_diff} -> {new_diff}) after {current_difficulty} question")
        
        return new_ability
    
    def estimate_ability_mle(self, responses: List[Dict]) -> float:
        """Estimate ability using Maximum Likelihood Estimation with adaptive learning rate"""
        
        if not responses:
            return self.default_ability
        
        def likelihood(ability):
            log_likelihood = 0
            for response in responses:
                prob = self.calculate_probability(ability, response['question_id'])
                if response['response']:
                    log_likelihood += np.log(max(prob, 1e-10))
                else:
                    log_likelihood += np.log(max(1 - prob, 1e-10))
            return -log_likelihood  # Negative for minimization
        
        # Use finer grid search for better precision
        abilities = np.linspace(self.ability_range[0], self.ability_range[1], 121)
        likelihoods = [likelihood(a) for a in abilities]
        
        best_ability = abilities[np.argmin(likelihoods)]
        
        # Apply adaptive smoothing based on number of responses
        if len(responses) < 5:
            # For early responses, move more conservatively toward estimated ability
            current_ability = self.get_student_ability(responses[0].get('student_id', ''))
            smoothing_factor = 0.3 + (len(responses) * 0.1)  # 0.3 to 0.7
            best_ability = current_ability + smoothing_factor * (best_ability - current_ability)
        
        # Constrain to reasonable range
        return max(self.ability_range[0], min(self.ability_range[1], best_ability))
    
    def get_student_ability(self, student_id: str) -> float:
        """Get current ability estimate for student"""
        if student_id in student_sessions:
            return student_sessions[student_id]['ability']
        return self.default_ability
    
    def prepare_question_for_serving(self, question: Dict) -> Dict:
        """Prepare question for API response, including image data if needed"""
        prepared_question = question.copy()
        
        # Add image data if question has an image
        if question.get('has_image') and question.get('image_id'):
            image_id = question['image_id']
            if image_id in self.image_mappings:
                image_info = self.image_mappings[image_id]
                prepared_question['image_url'] = f"/api/image/{image_id}"
                prepared_question['image_info'] = {
                    'id': image_id,
                    'format': self.get_image_format(image_info['full_path'])
                }
        
        return prepared_question
    
    def get_image_format(self, image_path: str) -> str:
        """Get image format from file extension"""
        ext = Path(image_path).suffix.lower()
        format_map = {
            '.png': 'PNG',
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.gif': 'GIF',
            '.bmp': 'BMP',
            '.webp': 'WEBP'
        }
        return format_map.get(ext, 'PNG')
    
    def get_assessment_summary(self, student_id: str) -> Dict:
        """Get comprehensive assessment summary for student"""
        if student_id not in student_sessions:
            return {'error': 'Student session not found'}
        
        session = student_sessions[student_id]
        responses = session['responses']
        
        if not responses:
            return {'error': 'No responses found'}
        
        # Calculate performance metrics
        total_questions = len(responses)
        correct_answers = sum(1 for r in responses if r['response'])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Topic-wise performance
        topic_performance = {}
        for response in responses:
            question_id = response['question_id']
            if question_id in self.questions:
                topic = self.questions[question_id]['topic']
                if topic not in topic_performance:
                    topic_performance[topic] = {'correct': 0, 'total': 0}
                topic_performance[topic]['total'] += 1
                if response['response']:
                    topic_performance[topic]['correct'] += 1
        
        # Add accuracy to topic performance
        for topic in topic_performance:
            topic_performance[topic]['accuracy'] = (
                topic_performance[topic]['correct'] / topic_performance[topic]['total']
            )
        
        # Difficulty-wise performance
        difficulty_performance = {}
        for response in responses:
            question_id = response['question_id']
            if question_id in self.questions:
                difficulty = self.questions[question_id]['difficulty']
                if difficulty not in difficulty_performance:
                    difficulty_performance[difficulty] = {'correct': 0, 'total': 0}
                difficulty_performance[difficulty]['total'] += 1
                if response['response']:
                    difficulty_performance[difficulty]['correct'] += 1
        
        # Add accuracy to difficulty performance
        for difficulty in difficulty_performance:
            difficulty_performance[difficulty]['accuracy'] = (
                difficulty_performance[difficulty]['correct'] / difficulty_performance[difficulty]['total']
            )
        
        # Calculate time spent
        time_spent_minutes = (session['last_update'] - session['start_time']).total_seconds() / 60
        
        return {
            'success': True,
            'student_id': student_id,
            'current_ability': session['ability'],
            'ability_history': session['ability_history'],
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'overall_accuracy': accuracy,
            'topic_performance': topic_performance,
            'difficulty_performance': difficulty_performance,
            'start_time': session['start_time'].isoformat(),
            'last_update': session['last_update'].isoformat(),
            'session_duration_minutes': time_spent_minutes,
            'final_results': {
                'final_score': round(accuracy * 100, 1),
                'ability_level': f"{session['ability']:.2f}",
                'questions_answered': total_questions,
                'time_spent_minutes': round(time_spent_minutes, 1)
            },
            'recommendations': self._generate_recommendations(topic_performance, difficulty_performance, session['ability'])
        }
    
    def _generate_recommendations(self, topic_performance: Dict, difficulty_performance: Dict, ability: float) -> List[str]:
        """Generate personalized learning recommendations as structured objects"""
        def rec(icon: str, title: str, description: str, priority: str) -> Dict[str, str]:
            return { 'icon': icon, 'title': title, 'description': description, 'priority': priority }

        recs: List[Dict[str, str]] = []

        # Analyze topic performance
        weak_topics = []
        strong_topics = []
        for topic, perf in topic_performance.items():
            if perf.get('accuracy', 0) < 0.6:
                weak_topics.append(topic)
            elif perf.get('accuracy', 0) >= 0.8:
                strong_topics.append(topic)

        # Analyze difficulty performance
        weak_difficulties = []
        for difficulty, perf in difficulty_performance.items():
            if perf.get('accuracy', 0) < 0.5:
                weak_difficulties.append(difficulty)

        # Ability-based guidance
        if ability < -1.0:
            recs.append(rec('🧱', 'Build strong foundations', 'Focus on fundamentals with Very Easy questions to gain confidence.', 'high'))
            recs.append(rec('➗', 'Practice basics regularly', 'Daily practice on basic arithmetic and algebra will help you progress steadily.', 'medium'))
        elif ability < 0:
            recs.append(rec('🎯', 'Consolidate core skills', 'Work on Easy to Moderate questions to build accuracy and speed.', 'medium'))
            recs.append(rec('📘', 'Review key concepts', 'Revisit fundamental concepts before attempting harder problems.', 'medium'))
        elif ability < 1.0:
            recs.append(rec('🚀', 'Increase challenge gradually', 'Attempt more Moderate questions and sprinkle in a few Difficult ones.', 'low'))
            recs.append(rec('⏱️', 'Time management', 'Practice timed quizzes to improve consistency and endurance.', 'low'))
        else:
            recs.append(rec('🏆', 'Advance to tougher sets', 'You are ready for more Difficult problems and competitive practice.', 'low'))
            recs.append(rec('🤝', 'Teach to learn', 'Explaining concepts to peers can reinforce your mastery.', 'low'))

        # Topic-specific guidance
        if weak_topics:
            recs.append(rec('📚', 'Focus topics', f"Allocate extra practice to: {', '.join(weak_topics[:3])}.", 'high'))
        if strong_topics:
            recs.append(rec('⭐', 'Leverage strengths', f"Keep sharpening: {', '.join(strong_topics[:2])}.", 'low'))

        # Difficulty-specific guidance
        if weak_difficulties:
            pretty = ', '.join(weak_difficulties)
            recs.append(rec('🧩', 'Difficulty focus', f"Spend more sessions on {pretty} questions to lift accuracy.", 'medium'))

        # Limit to 5 items max for UI
        return recs[:5]


@app.route('/api/student/personalized_report', methods=['POST'])
def generate_personalized_report():
    """Generate a Gemini-powered personalized report using aggregated diagnosis data.
    Request JSON: { profile_id? , student_name?, student_grade?, api_key? , model? }
    Response JSON: { success, report_markdown, strengths, weaknesses, learning_path }
    """
    if not trained_model:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
    except Exception:
        data = {}

    profile_id = data.get('profile_id')
    name = data.get('student_name') or data.get('studentName')
    grade = data.get('student_grade') or data.get('studentGrade')
    if not profile_id:
        if not (name and grade):
            return jsonify({'success': False, 'error': 'profile_id or (student_name and student_grade) required'}), 400
        profile_id = _compute_profile_id(name, grade)

    # Gather diagnosis (reuse logic similar to /api/student/diagnosis)
    hist = _load_history(profile_id)
    responses = list(hist.get('responses', []))
    live_sessions = [s for s in student_sessions.values() if s.get('profile_id') == profile_id]
    for sess in live_sessions:
        for r in sess.get('responses', []):
            qid = r.get('question_id')
            q = trained_model.questions.get(qid, {})
            responses.append({
                'session_id': 'live',
                'timestamp': r.get('timestamp', datetime.now().isoformat()),
                'question_id': qid,
                'topic': q.get('topic'),
                'difficulty': q.get('difficulty'),
                'is_correct': r.get('response', False),
                'ability_after': sess.get('ability')
            })

    # Aggregate
    topic_stats: Dict[str, Dict[str, int]] = {}
    diff_stats: Dict[str, Dict[str, int]] = {}
    ability_vals: List[float] = []
    for r in responses:
        topic = r.get('topic') or 'Unknown'
        diff = r.get('difficulty') or 'Unknown'
        ok = bool(r.get('is_correct'))
        topic_stats.setdefault(topic, {'correct': 0, 'total': 0})
        topic_stats[topic]['total'] += 1
        topic_stats[topic]['correct'] += 1 if ok else 0
        diff_stats.setdefault(diff, {'correct': 0, 'total': 0})
        diff_stats[diff]['total'] += 1
        diff_stats[diff]['correct'] += 1 if ok else 0
        if r.get('ability_after') is not None:
            ability_vals.append(float(r.get('ability_after')))

    def rate(d: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in d.items():
            total = max(1, v['total'])
            out[k] = { **v, 'accuracy': round((v['correct']/total)*100, 1) }
        return out

    per_topic = rate(topic_stats)
    per_difficulty = rate(diff_stats)
    overall_accuracy = round((sum(v['correct'] for v in topic_stats.values()) / max(1, sum(v['total'] for v in topic_stats.values()))) * 100, 1)
    current_ability = ability_vals[-1] if ability_vals else trained_model.default_ability

    # Prepare a fallback learning path from our rule-based engine
    fallback_recs = trained_model._generate_recommendations(per_topic, per_difficulty, current_ability)

    # Build prompt for Gemini AI-Powered Personalized Report
    api_key = os.environ.get('GEMINI_API_KEY')
    model_name = data.get('model') or 'gemini-pro'

    if not api_key:
        # Return a graceful fallback if no key
        basic = {
            'success': True,
            'provider': 'fallback',
            'report_markdown': (
                f"### Personalized Report (Fallback)\n"
                f"Overall accuracy: {overall_accuracy}%\n\n"
                f"- Current estimated ability: {round(current_ability, 2)}\n"
                f"- Strong topics: {', '.join([t for t, s in per_topic.items() if s['accuracy'] >= 80]) or '—'}\n"
                f"- Focus topics: {', '.join([t for t, s in per_topic.items() if s['accuracy'] < 60]) or '—'}\n\n"
                f"#### Recommended Next Steps\n"
            ),
            'learning_path': fallback_recs,
            'strengths': [t for t, s in per_topic.items() if s['accuracy'] >= 80],
            'weaknesses': [t for t, s in per_topic.items() if s['accuracy'] < 60],
            'overall_accuracy': overall_accuracy,
            'current_ability': round(current_ability, 2),
        }
        return jsonify(basic)

    # Try using Gemini/GenAI if the library is available, but fail gracefully to deterministic fallback
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Prefer querying supported models dynamically where available
        model_to_try = None
        try:
            # List models and pick first one that supports text generation (if API supports this)
            available = []
            try:
                # Some SDKs provide list_models(); wrap defensively
                available = [m.id for m in getattr(genai, 'list_models', lambda: [])()]
            except Exception:
                # fallback to known names if list_models is not present
                available = []

            # Candidate names (kept as a fallback list)
            candidates = ['gemini-pro', 'gemini-1.0-pro', 'models/gemini-pro', 'gemini-1.5-pro']
            # Prefer any candidate that appears in available; otherwise try candidates in order
            for c in candidates:
                if c in available:
                    model_to_try = c
                    break
            if not model_to_try and available:
                model_to_try = available[0]
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
            model_to_try = None

        # Compose the content for a generation attempt
        summary = {
            'overall_accuracy': overall_accuracy,
            'current_ability': round(current_ability, 2),
            'per_topic': per_topic,
            'per_difficulty': per_difficulty,
        }

        system_prompt = (
            "You are an expert learning coach for K-12 assessments. "
            "Generate an AI-Powered Personalized Report based on the student's performance data. "
            "Be concise, encouraging, and give clear next steps. Output JSON with keys: report_markdown, strengths, weaknesses, learning_path." 
        )

        prompt = f"SYSTEM:\n{system_prompt}\n\nSTUDENT_PERFORMANCE_DATA:\n{json.dumps(summary, ensure_ascii=False)}\n\nRespond only with JSON."

        text = None
        if model_to_try:
            try:
                model = genai.GenerativeModel(model_to_try)
                resp = model.generate_content(prompt)
                text = resp.text or ''
            except Exception as model_err:
                logger.warning(f"Model {model_to_try} generation failed: {model_err}")
                text = None

        # If generation returned something, try to parse JSON safely
        parsed = None
        if text:
            try:
                if '```' in text:
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    if start != -1 and end != -1:
                        text = text[start:end]
                parsed = json.loads(text)
            except Exception as e:
                logger.warning(f"Failed to parse model output as JSON: {e}")
                parsed = None

        # If anything fails, return deterministic fallback (always succeed)
        if not parsed or 'report_markdown' not in parsed:
            # Deterministic fallback report
            report_md = (
                f"### Personalized Report\n\n"
                f"Overall accuracy: {overall_accuracy}%\n\n"
                f"Estimated ability: {round(current_ability, 2)}\n\n"
                f"Top strengths: {', '.join([t for t, s in per_topic.items() if s['accuracy'] >= 80]) or '—'}\n\n"
                f"Focus areas: {', '.join([t for t, s in per_topic.items() if s['accuracy'] < 60]) or '—'}\n\n"
                "Recommended next steps:\n"
            )
            # Add short bullets from fallback_recs
            for rec in fallback_recs:
                report_md += f"- {rec.get('title','Practice')} — {rec.get('description','Practice more')}\n"

            response = {
                'success': True,
                'provider': 'fallback',
                'report_markdown': report_md,
                'learning_path': fallback_recs,
                'strengths': [t for t, s in per_topic.items() if s['accuracy'] >= 80],
                'weaknesses': [t for t, s in per_topic.items() if s['accuracy'] < 60],
                'overall_accuracy': overall_accuracy,
                'current_ability': round(current_ability, 2),
            }
            return jsonify(response)

        # If parsed looks good, merge fallback learning path if missing
        if 'learning_path' not in parsed or not parsed['learning_path']:
            parsed['learning_path'] = fallback_recs

        parsed.setdefault('overall_accuracy', overall_accuracy)
        parsed.setdefault('current_ability', round(current_ability, 2))
        return jsonify({ 'success': True, 'provider': model_to_try or 'gemini', **parsed })

    except ModuleNotFoundError:
        # GenAI SDK not installed — return deterministic fallback
        logger.warning('google-generativeai not installed; returning fallback personalized report')
        report_md = (
            f"### Personalized Report\n\n"
            f"Overall accuracy: {overall_accuracy}%\n\n"
            f"Estimated ability: {round(current_ability, 2)}\n\n"
            f"Top strengths: {', '.join([t for t, s in per_topic.items() if s['accuracy'] >= 80]) or '—'}\n\n"
            f"Focus areas: {', '.join([t for t, s in per_topic.items() if s['accuracy'] < 60]) or '—'}\n\n"
            "Recommended next steps:\n"
        )
        for rec in fallback_recs:
            report_md += f"- {rec.get('title','Practice')} — {rec.get('description','Practice more')}\n"

        return jsonify({
            'success': True,
            'provider': 'fallback',
            'report_markdown': report_md,
            'learning_path': fallback_recs,
            'strengths': [t for t, s in per_topic.items() if s['accuracy'] >= 80],
            'weaknesses': [t for t, s in per_topic.items() if s['accuracy'] < 60],
            'overall_accuracy': overall_accuracy,
            'current_ability': round(current_ability, 2),
        })


def load_trained_model(model_path: str = 'trained_adaptive_assessment_model.json'):
    """Load the trained adaptive assessment model"""
    global trained_model
    
    try:
        from pathlib import Path
        base = Path(model_path)
        fixed_candidate = base.with_name(base.stem + '_ai_fixed.json')
        chosen = fixed_candidate if fixed_candidate.exists() and 'ai_fixed' not in base.name else base
        with open(chosen, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        trained_model = AdaptiveAssessmentEngine(model_data)
        logger.info(f"Successfully loaded trained model from {str(chosen)}")
        logger.info(f"Model contains {len(model_data['questions'])} questions")
        logger.info(f"Model version: {model_data['model_metadata'].get('version', 'unknown')}")
        
        return True
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    mongodb_status = "connected" if students_collection is not None else "disconnected"
    student_stats = get_student_stats() if students_collection is not None else {}
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': trained_model is not None,
        'gemini_ready': bool(os.environ.get('GEMINI_API_KEY')),
        'mongodb_status': mongodb_status,
        'student_stats': student_stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/admin/students', methods=['GET'])
def get_all_students_endpoint():
    """Get all students data"""
    try:
        students = get_all_students()
        return jsonify({
            'success': True,
            'students': students,
            'total_count': len(students)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/student/<profile_id>', methods=['DELETE'])
def delete_student_endpoint(profile_id):
    """Delete a specific student's data"""
    try:
        success = delete_student_data(profile_id)
        if success:
            return jsonify({'success': True, 'message': f'Student {profile_id} deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Student not found or could not be deleted'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
def get_system_stats():
    """Get overall system statistics"""
    try:
        stats = get_student_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/sanitize/prewarm', methods=['POST'])
def admin_prewarm_sanitizer():
    if not trained_model:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    stats = _prewarm_sanitizer_all_questions()
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/admin/sanitize/<qid>', methods=['POST'])
def admin_sanitize_single(qid: str):
    """Question sanitization feature has been disabled"""
    return jsonify({'success': False, 'error': 'Question sanitization feature has been disabled'}), 501

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 404
    
    metadata = trained_model.model_data['model_metadata']
    return jsonify({
        'model_info': metadata,
        'total_questions': len(trained_model.questions),
        'total_topics': len(trained_model.topics),
        'questions_with_images': sum(1 for q in trained_model.questions.values() if q.get('has_image')),
        'available_topics': list(trained_model.topics.keys())
    })

@app.route('/api/student/start', methods=['POST'])
def start_assessment():
    """Start a new assessment session for a student"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    student_id = data.get('student_id')
    student_name = data.get('student_name') or data.get('studentName')
    student_grade = data.get('student_grade') or data.get('studentGrade')
    # Optional explicit profile_id to uniquely scope history per authenticated user
    provided_profile_id = data.get('profile_id')
    # Optional: max questions per assessment (default 20, clamp 1..100)
    try:
        max_questions = int(data.get('max_questions', 20))
    except Exception:
        max_questions = 20
    max_questions = max(1, min(100, max_questions))
    
    if not student_id:
        return jsonify({'error': 'student_id is required'}), 400
    
    # Initialize student session
    # Use provided profile_id if available; otherwise fall back to name+grade hash
    profile_id = provided_profile_id or _compute_profile_id(student_name or '', student_grade or '')
    student_sessions[student_id] = {
        'ability': trained_model.default_ability,
        'responses': [],
        'ability_history': [trained_model.default_ability],
        'start_time': datetime.now(),
        'last_update': datetime.now(),
        'max_questions': max_questions,
        'student_name': student_name,
        'student_grade': student_grade,
        'profile_id': profile_id
    }
    
    # Get first question
    first_question = trained_model.select_next_question(student_id)
    
    if not first_question:
        return jsonify({'error': 'No questions available'}), 500
    
    # Ensure profile file exists/update basic profile
    hist = _load_history(profile_id)
    hist['profile'] = {
        'name': student_name,
        'grade': student_grade,
        'profile_id': profile_id
    }
    hist.setdefault('sessions', {})
    hist['sessions'][student_id] = {
        'started_at': datetime.now().isoformat(),
        'max_questions': max_questions
    }
    _save_history(profile_id, hist)

    return jsonify({
        'message': 'Assessment started successfully',
        'student_id': student_id,
        'initial_ability': trained_model.default_ability,
        'first_question': first_question,
        'max_questions': max_questions,
        'profile_id': profile_id
    })

@app.route('/api/student/question', methods=['POST'])
def get_next_question():
    """Get the next question for a student"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    student_id = data.get('student_id')
    answered_questions = data.get('answered_questions', [])
    target_topic = data.get('topic')
    target_difficulty = data.get('difficulty')
    
    if not student_id:
        return jsonify({'error': 'student_id is required'}), 400
    
    # Enforce per-session question limit
    session = student_sessions.get(student_id, {})
    max_questions = session.get('max_questions', 20)
    answered_count = len(session.get('responses', []))
    if answered_count >= max_questions:
        return jsonify({'error': 'No more questions available (limit reached)'}), 404

    question = trained_model.select_next_question(
        student_id, answered_questions, target_topic, target_difficulty
    )
    
    if not question:
        return jsonify({'error': 'No more questions available'}), 404
    
    return jsonify({
        'question': question,
        'current_ability': trained_model.get_student_ability(student_id)
    })

@app.route('/api/student/submit', methods=['POST'])
def submit_answer():
    """Submit an answer and get feedback"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    student_id = data.get('student_id')
    question_id = data.get('question_id')
    student_answer = data.get('answer')
    
    if not all([student_id, question_id, student_answer]):
        return jsonify({'error': 'student_id, question_id, and answer are required'}), 400
    
    # Get the correct answer
    if question_id not in trained_model.questions:
        return jsonify({'error': 'Question not found'}), 404
    
    correct_answer = trained_model.questions[question_id]['answer']
    is_correct = student_answer.upper() == correct_answer.upper()
    
    # Update student ability
    new_ability = trained_model.update_student_ability(student_id, question_id, is_correct)
    
    # Prepare comprehensive progress data for frontend
    session = student_sessions.get(student_id, {})
    responses = session.get('responses', [])
    answered_count = len(responses)
    
    # Calculate current score
    correct_count = sum(1 for r in responses if r.get('response', False))
    current_score = round((correct_count / answered_count * 100)) if answered_count > 0 else 0
    
    # Calculate knowledge level
    if answered_count >= 3:
        recent_correct = sum(1 for r in responses[-3:] if r.get('response', False))
        recent_performance = recent_correct / min(3, len(responses[-3:]))
        overall_performance = correct_count / answered_count
        knowledge_level = 0.6 * recent_performance + 0.4 * overall_performance
    else:
        knowledge_level = correct_count / answered_count if answered_count > 0 else 0
    
    # Calculate streaks
    consecutive_correct = 0
    consecutive_incorrect = 0
    
    for r in reversed(responses):
        if r.get('response', False):
            if consecutive_incorrect == 0:
                consecutive_correct += 1
            else:
                break
        else:
            if consecutive_correct == 0:
                consecutive_incorrect += 1
            else:
                break
    
    updated_progress = {
        'questions_answered': answered_count,
        'ability_estimate': new_ability,
        'predicted_success_probability': 0.5 + (new_ability / 6),
        'current_score': current_score,
        'knowledge_level': knowledge_level,
        'consecutive_correct': consecutive_correct,
        'consecutive_incorrect': consecutive_incorrect
    }
    
    adaptation_info = {
        'new_ability': new_ability,
        'difficulty_change': trained_model.get_optimal_difficulty_for_ability(new_ability),
        'current_difficulty': trained_model.get_optimal_difficulty_for_ability(new_ability),
        'next_difficulty_hint': trained_model.get_optimal_difficulty_for_ability(new_ability)
    }
    
    # Record to history
    try:
        profile_id = session.get('profile_id')
        if profile_id:
            hist = _load_history(profile_id)
            hist.setdefault('responses', [])
            q = trained_model.questions.get(question_id, {})
            hist['responses'].append({
                'session_id': student_id,
                'timestamp': datetime.now().isoformat(),
                'question_id': question_id,
                'topic': q.get('topic'),
                'difficulty': q.get('difficulty'),
                'is_correct': is_correct,
                'ability_after': new_ability
            })
            # update sessions meta
            hist.setdefault('sessions', {})
            sess_meta = hist['sessions'].get(student_id, {})
            sess_meta['last_activity'] = datetime.now().isoformat()
            sess_meta['answered'] = (sess_meta.get('answered', 0) or 0) + 1
            hist['sessions'][student_id] = sess_meta
            _save_history(profile_id, hist)
    except Exception as e:
        logger.error(f"Failed to record history: {e}")

    return jsonify({
        'is_correct': is_correct,
        'correct_answer': correct_answer,
        'new_ability': new_ability,
        'feedback': 'Correct!' if is_correct else f'Incorrect. The correct answer was {correct_answer}.',
        'updated_progress': updated_progress,
        'adaptation_info': adaptation_info
    })

@app.route('/api/student/summary', methods=['GET'])
def get_student_summary():
    """Get comprehensive assessment summary for a student"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    student_id = request.args.get('student_id')
    
    if not student_id:
        return jsonify({'error': 'student_id is required'}), 400
    
    summary = trained_model.get_assessment_summary(student_id)
    
    return jsonify(summary)

@app.route('/api/student/diagnosis', methods=['GET'])
def get_student_diagnosis():
    """Aggregate strengths and weaknesses across sessions for a student profile.
    Expects either profile_id, or student_name and student_grade.
    """
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500

    profile_id = request.args.get('profile_id')
    name = request.args.get('student_name') or request.args.get('studentName')
    grade = request.args.get('student_grade') or request.args.get('studentGrade')
    if not profile_id:
        if not (name and grade):
            return jsonify({'error': 'profile_id or (student_name and student_grade) required'}), 400
        profile_id = _compute_profile_id(name, grade)

    hist = _load_history(profile_id)
    responses = hist.get('responses', [])
    # Include current live session responses if any
    live_sessions = [s for s in student_sessions.values() if s.get('profile_id') == profile_id]
    for sess in live_sessions:
        for r in sess.get('responses', []):
            qid = r.get('question_id')
            q = trained_model.questions.get(qid, {})
            responses.append({
                'session_id': 'live',
                'timestamp': r.get('timestamp', datetime.now().isoformat()),
                'question_id': qid,
                'topic': q.get('topic'),
                'difficulty': q.get('difficulty'),
                'is_correct': r.get('response', False),
                'ability_after': sess.get('ability')
            })

    # Aggregate per-topic and per-difficulty
    topic_stats = {}
    diff_stats = {}
    ability_trend = []
    sessions_trend = {}

    for r in responses:
        topic = r.get('topic') or 'Unknown'
        diff = r.get('difficulty') or 'Unknown'
        is_correct = bool(r.get('is_correct'))
        ability_after = r.get('ability_after')
        session_id = r.get('session_id', 'unknown')

        s = topic_stats.setdefault(topic, {'correct': 0, 'total': 0})
        s['total'] += 1
        s['correct'] += 1 if is_correct else 0

        d = diff_stats.setdefault(diff, {'correct': 0, 'total': 0})
        d['total'] += 1
        d['correct'] += 1 if is_correct else 0

        if ability_after is not None:
            ability_trend.append({'t': r.get('timestamp'), 'ability': ability_after})

        sess = sessions_trend.setdefault(session_id, {'correct': 0, 'total': 0})
        sess['total'] += 1
        sess['correct'] += 1 if is_correct else 0

    # Prepare output
    def _rate(obj):
        out = {}
        for k, v in obj.items():
            acc = round((v['correct'] / v['total'] * 100), 1) if v['total'] else 0.0
            out[k] = {**v, 'accuracy': acc}
        return out

    result = {
        'profile': hist.get('profile', {'profile_id': profile_id}),
        'per_topic': _rate(topic_stats),
        'per_difficulty': _rate(diff_stats),
        'ability_trend': ability_trend,
        'sessions_trend': {k: {**v, 'accuracy': round((v['correct']/v['total']*100),1) if v['total'] else 0.0} for k, v in sessions_trend.items()},
        'total_responses': len(responses),
        'overall_accuracy': round(sum(topic_stats[k]['correct'] for k in topic_stats) / sum(topic_stats[k]['total'] for k in topic_stats) * 100, 1) if any(topic_stats[k]['total'] > 0 for k in topic_stats) else 0.0,
        'current_ability': ability_trend[-1]['ability'] if ability_trend else 0.0
    }

    return jsonify({'success': True, 'diagnosis': result})

@app.route('/api/image/<image_id>', methods=['GET'])
def serve_image(image_id):
    """Serve question images with improved mapping"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    logger.info(f"Looking for image: {image_id}")
    
    # First try the trained model's image mappings
    if image_id in trained_model.image_mappings:
        image_info = trained_model.image_mappings[image_id]
        image_path = Path(image_info['full_path'])
        
        logger.info(f"Found in mappings: {image_path}")
        
        if image_path.exists():
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type:
                mime_type = 'image/png'
            
            try:
                logger.info(f"SUCCESS: Serving mapped image: {image_path}")
                return send_file(str(image_path), mimetype=mime_type)
            except Exception as e:
                logger.error(f"Error serving mapped image {image_id}: {e}")
    
    # Extract question number from image_id
    question_num = None
    if image_id.startswith('img_'):
        try:
            question_num = image_id.replace('img_', '')
            logger.info(f"Extracted question number: {question_num}")
        except:
            logger.error(f"Could not extract question number from {image_id}")
            return jsonify({'error': 'Invalid image ID format'}), 400
    
    # Build comprehensive list of possible image paths for this question
    possible_image_paths = []
    
    if question_num:
        # Primary paths based on question number
        base_paths = [
            f"data/Geometry/Moderate_hard/Geometry(Moderate+Difficult)/Q{question_num}.png",
            f"data/Geometry/Easy_veryeasy/Geometry(easy+very easy)/Q{question_num}.png",
            f"data/Geometry/Coordinate Geometry/Q{question_num}.png",
        ]
        
        # Add variations for compound questions (Q32&33, Q47&Q48, etc.)
        compound_variations = [
            f"data/Geometry/Moderate_hard/Geometry(Moderate+Difficult)/Q{question_num}&{int(question_num)+1}.png",
            f"data/Geometry/Moderate_hard/Geometry(Moderate+Difficult)/Q{int(question_num)-1}&{question_num}.png",
            f"data/Geometry/Easy_veryeasy/Geometry(easy+very easy)/Q{question_num}&{int(question_num)+1}.png",
            f"data/Geometry/Easy_veryeasy/Geometry(easy+very easy)/Q{int(question_num)-1}&{question_num}.png",
        ]
        
        possible_image_paths.extend(base_paths)
        
        # Only add compound variations if question_num is numeric
        try:
            int(question_num)
            possible_image_paths.extend(compound_variations)
        except ValueError:
            pass
    
    # Try each possible path
    for path_str in possible_image_paths:
        image_path = Path(path_str)
        logger.info(f"Trying path: {image_path}")
        
        if image_path.exists():
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type:
                mime_type = 'image/png'
            
            try:
                logger.info(f"SUCCESS: Found and serving image: {image_path}")
                return send_file(str(image_path), mimetype=mime_type)
            except Exception as e:
                logger.error(f"Error serving image from {image_path}: {e}")
                continue
    
    # If still not found, search the questions in the model for this specific question
    if hasattr(trained_model, 'adaptive_engine') and trained_model.adaptive_engine:
        for qid, question in trained_model.adaptive_engine.questions.items():
            question_id_from_model = str(question.get('id', ''))
            
            if question_id_from_model == question_num:
                logger.info(f"Found question {qid} matching ID {question_num}")
                
                # Check if question has specific image information
                image_path_str = question.get('image_path')
                if image_path_str and image_path_str != 'None':
                    image_path = Path(image_path_str)
                    logger.info(f"Question specifies image path: {image_path}")
                    
                    if image_path.exists():
                        mime_type, _ = mimetypes.guess_type(str(image_path))
                        if not mime_type:
                            mime_type = 'image/png'
                        
                        try:
                            logger.info(f"SUCCESS: Serving image from question metadata: {image_path}")
                            return send_file(str(image_path), mimetype=mime_type)
                        except Exception as e:
                            logger.error(f"Error serving image from question metadata: {e}")
                
                break
    
    # If we reach here, the image was not found
    logger.error(f"Image not found for {image_id} (question {question_num})")
    logger.error(f"Searched paths: {possible_image_paths}")
    
    # DO NOT serve any fallback test image - return proper 404
    return jsonify({
        'error': 'Image not found', 
        'image_id': image_id,
        'searched_paths': possible_image_paths
    }), 404

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get all available topics and their statistics"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'topics': trained_model.topics
    })

@app.route('/api/questions/search', methods=['POST'])
def search_questions():
    """Search questions by criteria"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    topic = data.get('topic')
    difficulty = data.get('difficulty')
    has_image = data.get('has_image')
    limit = data.get('limit', 10)
    
    questions = list(trained_model.questions.values())
    
    # Apply filters
    if topic:
        questions = [q for q in questions if q['topic'].lower() == topic.lower()]
    
    if difficulty:
        questions = [q for q in questions if q['difficulty'].lower() == difficulty.lower()]
    
    if has_image is not None:
        questions = [q for q in questions if q.get('has_image') == has_image]
    
    # Limit results
    questions = questions[:limit]
    
    # Prepare questions for serving
    prepared_questions = [
        trained_model.prepare_question_for_serving(q) for q in questions
    ]
    
    return jsonify({
        'questions': prepared_questions,
        'total_found': len(prepared_questions)
    })

# ==============================================
# FRONTEND COMPATIBILITY ENDPOINTS
# ==============================================
# These endpoints maintain compatibility with the existing frontend

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment_compatible():
    """Frontend-compatible start assessment endpoint"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    # Extract name and grade from frontend
    student_name = data.get('studentName', 'Unknown')
    student_grade = data.get('studentGrade', 'Unknown')
    
    # Create unique student ID
    student_id = f"{student_name}_{student_grade}_{int(datetime.now().timestamp() * 1000)}"
    
    # Initialize student session
    student_sessions[student_id] = {
        'ability': trained_model.default_ability,
        'responses': [],
        'ability_history': [trained_model.default_ability],
        'answered_questions': [],  # Track answered questions to avoid repetition (using list instead of set)
        'start_time': datetime.now(),
        'last_update': datetime.now(),
        'name': student_name,
        'grade': student_grade
    }
    
    # Get first question
    first_question = trained_model.select_next_question(student_id)
    
    if not first_question:
        return jsonify({'error': 'No questions available'}), 500
    
    return jsonify({
        'success': True,
        'session_id': student_id,
        'starting_difficulty': 'Very Easy',
        'message': 'Assessment started successfully'
    })

@app.route('/api/get-question', methods=['POST'])
def get_question_compatible():
    """Frontend-compatible get question endpoint"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'success': False, 'error': 'session_id is required'}), 400
    
    if session_id not in student_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 400
    
    # Get answered questions from both responses and tracking list
    session = student_sessions[session_id]
    answered_from_responses = [r['question_id'] for r in session.get('responses', [])]
    answered_from_list = session.get('answered_questions', [])
    answered_questions = list(set(answered_from_responses + answered_from_list))  # Combine and deduplicate
    
    # Check if assessment should complete (e.g., after 20 questions)
    if len(answered_questions) >= 20:
        return jsonify({
            'success': False,
            'assessment_complete': True,
            'message': 'Assessment completed'
        })
    
    # Get next question
    question = trained_model.select_next_question(session_id, answered_questions)
    
    if not question:
        return jsonify({
            'success': False,
            'assessment_complete': True,
            'message': 'No more questions available'
        })
    
    # Convert to frontend format
    frontend_question = {
        'id': question['id'],
        'question_text': question['text'],
        'option_a': question['options'].get('A', ''),
        'option_b': question['options'].get('B', ''),
        'option_c': question['options'].get('C', ''),
        'option_d': question['options'].get('D', ''),
        'answer': question['answer'].lower(),
        'difficulty': question['difficulty'],
        'topic': question.get('topic', 'General'),
        'image_path': question.get('image_path'),
        'has_image': question.get('has_image', False)
    }
    
    # Create comprehensive progress info
    current_ability = session['ability']
    responses = session.get('responses', [])
    answered_count = len(answered_questions)
    
    # Calculate current score
    correct_count = sum(1 for r in responses if r.get('response', False))
    current_score = round((correct_count / answered_count * 100)) if answered_count > 0 else 0
    
    # Calculate knowledge level
    if answered_count >= 3:
        recent_correct = sum(1 for r in responses[-3:] if r.get('response', False))
        recent_performance = recent_correct / min(3, len(responses[-3:]))
        overall_performance = correct_count / answered_count
        knowledge_level = 0.6 * recent_performance + 0.4 * overall_performance
    else:
        knowledge_level = correct_count / answered_count if answered_count > 0 else 0
    
    # Calculate streaks
    consecutive_correct = 0
    consecutive_incorrect = 0
    
    for r in reversed(responses):
        if r.get('response', False):
            if consecutive_incorrect == 0:
                consecutive_correct += 1
            else:
                break
        else:
            if consecutive_correct == 0:
                consecutive_incorrect += 1
            else:
                break
    
    progress = {
        'questions_answered': answered_count,
        'predicted_success_probability': 0.5 + (current_ability / 6),  # Normalize to 0-1
        'current_difficulty': question['difficulty'],
        'ability_estimate': current_ability,
        'current_score': current_score,
        'knowledge_level': knowledge_level,
        'consecutive_correct': consecutive_correct,
        'consecutive_incorrect': consecutive_incorrect
    }
    
    return jsonify({
        'success': True,
        'question': frontend_question,
        'student_progress': progress
    })

@app.route('/api/submit-response', methods=['POST'])
def submit_response_compatible():
    """Frontend-compatible submit response endpoint"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    selected_option = data.get('selected_option')
    is_correct = data.get('is_correct')
    
    if not all([session_id, question_id, selected_option is not None]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    if session_id not in student_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 400
    
    # Convert selected option to match our format
    if selected_option in ['a', 'b', 'c', 'd']:
        selected_option = selected_option.upper()
    
    # Get correct answer and validate
    question = trained_model.questions.get(question_id)
    if not question:
        return jsonify({'success': False, 'error': 'Question not found'}), 404
    
    correct_answer = question['answer']
    actual_is_correct = selected_option == correct_answer
    
    # Update student ability
    new_ability = trained_model.update_student_ability(session_id, question_id, actual_is_correct)
    
    # Prepare response with comprehensive progress data
    session = student_sessions[session_id]
    responses = session.get('responses', [])
    answered_count = len(responses)
    
    feedback_text = "Correct! Well done." if actual_is_correct else f"Incorrect. The correct answer was {correct_answer}."
    
    # Calculate current score
    correct_count = sum(1 for r in responses if r.get('response', False))
    current_score = round((correct_count / answered_count * 100)) if answered_count > 0 else 0
    
    # Calculate knowledge level (recent performance weighted)
    if answered_count >= 3:
        recent_correct = sum(1 for r in responses[-3:] if r.get('response', False))
        recent_performance = recent_correct / min(3, len(responses[-3:]))
        overall_performance = correct_count / answered_count
        knowledge_level = 0.6 * recent_performance + 0.4 * overall_performance
    else:
        knowledge_level = correct_count / answered_count if answered_count > 0 else 0
    
    # Calculate streaks
    consecutive_correct = 0
    consecutive_incorrect = 0
    
    # Count current streak from the end
    for r in reversed(responses):
        if r.get('response', False):
            if consecutive_incorrect == 0:  # Still in correct streak
                consecutive_correct += 1
            else:
                break
        else:
            if consecutive_correct == 0:  # Still in incorrect streak
                consecutive_incorrect += 1
            else:
                break
    
    progress = {
        'questions_answered': answered_count,
        'ability_estimate': new_ability,
        'predicted_success_probability': 0.5 + (new_ability / 6),
        'current_score': current_score,
        'knowledge_level': knowledge_level,
        'consecutive_correct': consecutive_correct,
        'consecutive_incorrect': consecutive_incorrect
    }
    
    adaptation_info = {
        'new_ability': new_ability,
        'difficulty_change': trained_model.get_optimal_difficulty_for_ability(new_ability)
    }
    
    return jsonify({
        'success': True,
        'is_correct': actual_is_correct,
        'correct_answer': correct_answer,
        'feedback': feedback_text,
        'updated_progress': progress,
        'adaptation_info': adaptation_info
    })

@app.route('/api/get-assessment-results', methods=['POST'])
def get_assessment_results_compatible():
    """Frontend-compatible get results endpoint"""
    if not trained_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    session_id = data.get('session_id')
    
    if not session_id or session_id not in student_sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 400
    
    # Get assessment summary from our enhanced system
    try:
        summary_response = get_student_summary()
        if hasattr(summary_response, 'get_json'):
            summary = summary_response.get_json()
        else:
            summary = summary_response
        
        session = student_sessions[session_id]
        responses = session.get('responses', [])
        
        # Calculate statistics
        total_questions = len(responses)
        correct_answers = sum(1 for r in responses if r.get('response', False))
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        # Prepare results in frontend format
        results = {
            'success': True,
            'student_name': session.get('name', 'Student'),
            'student_grade': session.get('grade', 'Unknown'),
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy_percentage': round(accuracy, 1),
            'final_ability': session['ability'],
            'ability_level': trained_model.get_optimal_difficulty_for_ability(session['ability']),
            'time_taken': str(datetime.now() - session['start_time']).split('.')[0],
            'detailed_performance': summary if 'error' not in summary else {}
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error generating assessment results: {e}")
        # Fallback response
        session = student_sessions[session_id]
        responses = session.get('responses', [])
        total_questions = len(responses)
        correct_answers = sum(1 for r in responses if r.get('response', False))
        
        return jsonify({
            'success': True,
            'student_name': session.get('name', 'Student'),
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy_percentage': round((correct_answers / max(total_questions, 1)) * 100, 1),
            'final_ability': session['ability'],
            'time_taken': str(datetime.now() - session['start_time']).split('.')[0]
        })


@app.route('/api/train-model', methods=['POST'])
def train_model_endpoint():
    """Endpoint to retrain the adaptive assessment model"""
    try:
        logger.info("Starting model training...")
        
        # Import training functionality
        import sys
        import importlib.util
        
        # Load the trainer module
        spec = importlib.util.spec_from_file_location("trainer", "train_adaptive_model.py")
        trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer_module)
        
        # Create trainer instance and train
        trainer = trainer_module.AdaptiveAssessmentTrainer()
        model_data = trainer.train_model()
        trainer.save_model('trained_adaptive_assessment_model.json')
        
        # Reload the trained model in the server
        global trained_model
        if load_trained_model():
            logger.info("Model retrained and reloaded successfully!")
            return jsonify({
                'success': True,
                'message': 'Model trained and reloaded successfully',
                'questions_count': len(model_data.get('questions', [])),
                'topics_count': len(model_data.get('topics', {})),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Training completed but failed to reload model'
            }), 500
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Training failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Load the trained model on startup
    _load_env_from_dotenv()
    if load_trained_model():
        logger.info("Starting adaptive assessment API server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Please run train_adaptive_model.py first.")
        print("Please run: python train_adaptive_model.py")