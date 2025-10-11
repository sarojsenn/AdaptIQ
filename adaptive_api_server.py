import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Global variables
questions_db = None
student_sessions = {}

def generate_missing_options(question):
    """Generate plausible options for questions that don't have them or have incomplete options"""
    
    # Count how many options are missing or empty
    options = ['option_a', 'option_b', 'option_c', 'option_d']
    empty_options = []
    existing_options = []
    
    for i, opt in enumerate(options):
        if not question[opt] or question[opt] in ['', 'nan', 'None']:
            empty_options.append(opt)
        else:
            existing_options.append((opt, question[opt]))
    
    # If most options are missing, generate a complete set
    if len(empty_options) >= 3:
        question_text = question['question_text'].lower()
        
        if 'percent' in question_text and 'day' in question_text and '3 hours' in question_text:
            # Specific case: What percent of day is 3 hours?
            # Answer: 3/24 * 100 = 12.5%
            question['option_a'] = '12.5%'
            question['option_b'] = '10%'
            question['option_c'] = '15%'
            question['option_d'] = '20%'
            
        elif '270 candidate' in question_text and 'examination' in question_text:
            # Specific case: 270 candidates, 252 passed = 252/270 * 100 = 93.33%
            # Only fill missing options, preserve existing ones
            if not question['option_a'] or question['option_a'] in ['', 'nan']:
                question['option_a'] = '85%'
            if not question['option_b'] or question['option_b'] in ['', 'nan']:
                question['option_b'] = '90%'
            if not question['option_c'] or question['option_c'] in ['', 'nan']:
                question['option_c'] = '92%'
            if not question['option_d'] or question['option_d'] in ['', 'nan']:
                question['option_d'] = '93.33%'
            
        elif 'sulphur' in question_text and '2250' in question_text:
            # 5 out of 2250 = 5/2250 * 100 = 0.22%
            question['option_a'] = '0.20%'
            question['option_b'] = '0.22%'
            question['option_c'] = '0.25%'
            question['option_d'] = '0.30%'
            
        elif 'percent' in question_text:
            # General percentage questions - generate reasonable options
            if question['answer'] == 'a':
                question['option_a'] = '12.5%'
                question['option_b'] = '25%'
                question['option_c'] = '50%'
                question['option_d'] = '75%'
            elif question['answer'] == 'b':
                question['option_a'] = '10%'
                question['option_b'] = '20%'
                question['option_c'] = '30%'
                question['option_d'] = '40%'
            elif question['answer'] == 'c':
                question['option_a'] = '15%'
                question['option_b'] = '25%'
                question['option_c'] = '35%'
                question['option_d'] = '45%'
            elif question['answer'] == 'd':
                question['option_a'] = '60%'
                question['option_b'] = '70%'
                question['option_c'] = '80%'
                question['option_d'] = '90%'
        else:
            # Generic numerical options
            if question['answer'] == 'a':
                question['option_a'] = '12'
                question['option_b'] = '15'
                question['option_c'] = '18'
                question['option_d'] = '20'
            elif question['answer'] == 'b':
                question['option_a'] = '10'
                question['option_b'] = '25'
                question['option_c'] = '30'
                question['option_d'] = '35'
            elif question['answer'] == 'c':
                question['option_a'] = '20'
                question['option_b'] = '25'
                question['option_c'] = '30'
                question['option_d'] = '35'
            elif question['answer'] == 'd':
                question['option_a'] = '40'
                question['option_b'] = '45'
                question['option_c'] = '50'
                question['option_d'] = '55'
    
    # Handle cases where we have some options but not all
    elif len(empty_options) > 0 and len(existing_options) > 0:
        # Try to generate similar options based on existing ones
        existing_values = [opt[1] for opt in existing_options]
        question_text = question['question_text'].lower()
        
        # Special handling for specific questions
        if '270 candidate' in question_text and 'examination' in question_text:
            # For 270 candidates question, the correct answer should be 93.33%
            percentage_options = ['75%', '80%', '85%', '90%', '93.33%', '95%']
            for opt in empty_options:
                for val in percentage_options:
                    if val not in existing_values:
                        question[opt] = val
                        existing_values.append(val)
                        break
        
        # If we have percentage values, generate more percentage options
        elif any('%' in str(val) for val in existing_values):
            # Extract numeric values from existing percentages to generate smart options
            existing_nums = []
            for val in existing_values:
                if '%' in str(val):
                    try:
                        num = float(str(val).replace('%', ''))
                        existing_nums.append(num)
                    except:
                        pass
            
            if existing_nums:
                # Generate options around the existing values
                base_num = max(existing_nums) if existing_nums else 80
                if base_num < 50:
                    percentage_options = ['10%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']
                else:
                    percentage_options = ['75%', '80%', '85%', '90%', '93.33%', '95%', '100%']
            else:
                percentage_options = ['75%', '80%', '85%', '90%', '93.33%', '95%']
            
            # Fill empty options with values not already used
            for opt in empty_options:
                for val in percentage_options:
                    if val not in existing_values:
                        question[opt] = val
                        existing_values.append(val)
                        break
        
        # If we have numerical values, generate similar numerical options
        elif any(str(val).replace('.', '').replace('-', '').isdigit() for val in existing_values):
            base_values = ['10', '15', '20', '25', '30', '35', '40', '45', '50']
            for opt in empty_options:
                for val in base_values:
                    if val not in existing_values:
                        question[opt] = val
                        existing_values.append(val)
                        break
    
    return question

def load_questions_from_csv():
    """Load and clean questions from the actual CSV dataset"""
    global questions_db
    
    try:
        # Try different encodings to read the CSV
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                print(f"Trying to load CSV with {encoding} encoding...")
                df = pd.read_csv('data/questions.csv', encoding=encoding)  # Don't skip header
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except (UnicodeDecodeError, Exception) as e:
                print(f"Failed with {encoding}: {e}")
                continue
        
        if df is None:
            raise Exception("Could not read CSV with any encoding")
        
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        # Clean the data more carefully
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Check if 'id' column exists and has valid data
        if 'id' not in df.columns:
            raise Exception(f"'id' column not found. Available columns: {df.columns.tolist()}")
        
        # Clean and filter by id
        df = df.dropna(subset=['id'])
        
        # Convert id to string first, then check if it's numeric
        df['id'] = df['id'].astype(str)
        df = df[df['id'].str.replace('.0', '').str.isdigit()]
        df['id'] = df['id'].str.replace('.0', '').astype(int)
        
        print(f"After ID cleaning: {len(df)} rows")
        
        # Clean other columns
        required_columns = ['question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'answer', 'difficulty']
        for col in required_columns:
            if col not in df.columns:
                raise Exception(f"Required column '{col}' not found")
        
        # Clean difficulty column
        df['difficulty'] = df['difficulty'].astype(str).str.strip()
        df = df[df['difficulty'].notna()]
        df = df[df['difficulty'] != 'nan']
        
        print(f"After difficulty cleaning: {len(df)} rows")
        print(f"Unique difficulties: {df['difficulty'].unique()}")
        
        # Clean question text - remove rows with missing questions
        df = df.dropna(subset=['question_text'])
        df = df[df['question_text'].astype(str).str.strip() != '']
        df = df[df['question_text'].astype(str) != 'nan']
        
        print(f"After question text cleaning: {len(df)} rows")
        
        # Fill missing options with empty strings
        for col in ['option_a', 'option_b', 'option_c', 'option_d']:
            df[col] = df[col].fillna('')
        
        # Clean answer column
        df['answer'] = df['answer'].astype(str).str.lower().str.strip()
        df = df[df['answer'].isin(['a', 'b', 'c', 'd'])]
        
        print(f"After answer cleaning: {len(df)} rows")
        
        # Check which questions will need option generation
        missing_options_count = 0
        incomplete_options_count = 0
        
        for _, row in df.iterrows():
            empty_count = 0
            for opt_col in ['option_a', 'option_b', 'option_c', 'option_d']:
                if pd.isna(row[opt_col]) or str(row[opt_col]).strip() == '' or str(row[opt_col]).strip() == 'nan':
                    empty_count += 1
            
            if empty_count == 4:
                missing_options_count += 1
            elif empty_count > 0:
                incomplete_options_count += 1
        
        print(f"Questions with completely missing options: {missing_options_count}")
        print(f"Questions with incomplete options: {incomplete_options_count}")
        print(f"Total questions needing option generation: {missing_options_count + incomplete_options_count}")
        
        # Convert to list of dictionaries and handle missing options
        questions_list = []
        for _, row in df.iterrows():
            question = {
                'id': int(row['id']),
                'question_text': str(row['question_text']).strip(),
                'option_a': str(row['option_a']).strip() if pd.notna(row['option_a']) and str(row['option_a']).strip() else '',
                'option_b': str(row['option_b']).strip() if pd.notna(row['option_b']) and str(row['option_b']).strip() else '',
                'option_c': str(row['option_c']).strip() if pd.notna(row['option_c']) and str(row['option_c']).strip() else '',
                'option_d': str(row['option_d']).strip() if pd.notna(row['option_d']) and str(row['option_d']).strip() else '',
                'answer': str(row['answer']).lower().strip(),
                'difficulty': str(row['difficulty']).strip(),
                'tags': str(row.get('tags', 'General')).strip()
            }
            
            # Generate options for questions that don't have them but should
            question = generate_missing_options(question)
            questions_list.append(question)
        
        # Group questions by difficulty
        questions_db = {
            'Very easy': [],
            'Easy': [],
            'Moderate': [],
            'Difficult': []
        }
        
        for question in questions_list:
            difficulty = question['difficulty']
            if difficulty in questions_db:
                questions_db[difficulty].append(question)
            else:
                # Handle any variations in difficulty names
                if 'easy' in difficulty.lower():
                    if 'very' in difficulty.lower():
                        questions_db['Very easy'].append(question)
                    else:
                        questions_db['Easy'].append(question)
                elif 'moderate' in difficulty.lower():
                    questions_db['Moderate'].append(question)
                elif 'difficult' in difficulty.lower():
                    questions_db['Difficult'].append(question)
        
        total_questions = sum(len(questions) for questions in questions_db.values())
        
        print(f"✅ Loaded {total_questions} questions from dataset:")
        for difficulty, questions in questions_db.items():
            print(f"  - {difficulty}: {len(questions)} questions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading questions: {e}")
        return False

def get_student_difficulty_level(student_performance):
    """Determine appropriate difficulty level based on student performance"""
    if not student_performance['responses']:
        return 'Very easy'  # Start with very easy
    
    # Calculate recent performance (last 3 questions)
    recent_responses = student_performance['responses'][-3:]
    recent_correct = sum(1 for r in recent_responses if r['is_correct'])
    recent_accuracy = recent_correct / len(recent_responses)
    
    # Calculate overall performance
    total_correct = sum(1 for r in student_performance['responses'] if r['is_correct'])
    overall_accuracy = total_correct / len(student_performance['responses'])
    
    # Weight recent performance more heavily
    weighted_accuracy = 0.7 * recent_accuracy + 0.3 * overall_accuracy
    
    # Determine next difficulty level
    current_level = student_performance.get('current_difficulty', 'Very easy')
    
    # Adaptive logic based on performance
    if weighted_accuracy >= 0.8:  # 80%+ accuracy - increase difficulty
        if current_level == 'Very easy':
            return 'Easy'
        elif current_level == 'Easy':
            return 'Moderate'
        elif current_level == 'Moderate':
            return 'Difficult'
        else:
            return 'Difficult'  # Stay at highest level
    
    elif weighted_accuracy >= 0.6:  # 60-79% accuracy - maintain or slightly adjust
        return current_level
    
    else:  # <60% accuracy - decrease difficulty
        if current_level == 'Difficult':
            return 'Moderate'
        elif current_level == 'Moderate':
            return 'Easy'
        elif current_level == 'Easy':
            return 'Very easy'
        else:
            return 'Very easy'  # Stay at lowest level

def select_next_question(session):
    """Select the next question based on adaptive logic"""
    student_performance = session
    target_difficulty = get_student_difficulty_level(student_performance)
    
    # Update current difficulty
    student_performance['current_difficulty'] = target_difficulty
    
    # Get available questions at target difficulty
    available_questions = [
        q for q in questions_db[target_difficulty] 
        if q['id'] not in student_performance['answered_questions']
    ]
    
    # If no questions at target difficulty, try adjacent levels
    if not available_questions:
        difficulty_order = ['Very easy', 'Easy', 'Moderate', 'Difficult']
        current_index = difficulty_order.index(target_difficulty)
        
        # Try easier first, then harder
        for offset in [-1, 1, -2, 2]:
            new_index = current_index + offset
            if 0 <= new_index < len(difficulty_order):
                alt_difficulty = difficulty_order[new_index]
                available_questions = [
                    q for q in questions_db[alt_difficulty] 
                    if q['id'] not in student_performance['answered_questions']
                ]
                if available_questions:
                    target_difficulty = alt_difficulty
                    break
    
    if not available_questions:
        return None, target_difficulty
    
    # Select a random question from available ones
    # In a more sophisticated system, you could rank by information value
    selected_question = np.random.choice(available_questions)
    
    return selected_question, target_difficulty

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if questions_db:
        total_questions = sum(len(questions) for questions in questions_db.values())
        return jsonify({
            'status': 'healthy',
            'questions_loaded': True,
            'total_questions': total_questions,
            'questions_by_difficulty': {k: len(v) for k, v in questions_db.items()}
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'questions_loaded': False,
            'error': 'Questions not loaded'
        }), 500

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment():
    """Start a new assessment session"""
    try:
        data = request.json
        student_name = data.get('studentName')
        student_grade = data.get('studentGrade')
        
        if not student_name or not student_grade:
            return jsonify({'error': 'Student name and grade are required'}), 400
        
        # Create new session
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        # Initialize session
        student_sessions[session_id] = {
            'student_name': student_name,
            'student_grade': student_grade,
            'start_time': datetime.now(),
            'current_question': 0,
            'responses': [],
            'answered_questions': set(),
            'current_difficulty': 'Very easy',  # Start with very easy
            'ability_estimate': 0.0,
            'consecutive_correct': 0,
            'consecutive_incorrect': 0
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'Assessment session started for {student_name}',
            'starting_difficulty': 'Very easy'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start assessment: {str(e)}'}), 500

@app.route('/api/get-question', methods=['POST'])
def get_next_question():
    """Get the next adaptive question for a student"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in student_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = student_sessions[session_id]
        
        # Check if assessment should end
        if len(session['responses']) >= 10:  # Limit to 10 questions
            return jsonify({
                'success': False,
                'message': 'Assessment complete',
                'assessment_complete': True
            })
        
        # Select next question using adaptive logic
        next_question, target_difficulty = select_next_question(session)
        
        if next_question is None:
            return jsonify({
                'success': False,
                'message': 'No more questions available',
                'assessment_complete': True
            })
        
        # Calculate predicted probability based on difficulty and past performance
        if session['responses']:
            recent_accuracy = sum(1 for r in session['responses'][-3:] if r['is_correct']) / min(3, len(session['responses']))
            difficulty_adjustment = {
                'Very easy': 0.3,
                'Easy': 0.1,
                'Moderate': -0.1,
                'Difficult': -0.3
            }
            predicted_prob = max(0.2, min(0.8, recent_accuracy + difficulty_adjustment.get(target_difficulty, 0)))
        else:
            predicted_prob = 0.7  # Start optimistic for very easy questions
        
        return jsonify({
            'success': True,
            'question': next_question,
            'student_progress': {
                'current_difficulty': target_difficulty,
                'questions_answered': len(session['responses']),
                'predicted_success_probability': round(predicted_prob, 2),
                'ability_estimate': round(session['ability_estimate'], 2)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get question: {str(e)}'}), 500

@app.route('/api/submit-response', methods=['POST'])
def submit_response():
    """Submit a student's response and update the adaptive model"""
    try:
        data = request.json
        session_id = data.get('session_id')
        question_id = data.get('question_id')
        selected_option = data.get('selected_option')
        is_correct = data.get('is_correct')
        
        if session_id not in student_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = student_sessions[session_id]
        
        # Record response
        response_data = {
            'question_id': question_id,
            'selected_option': selected_option,
            'is_correct': is_correct,
            'difficulty': session['current_difficulty'],
            'timestamp': datetime.now().isoformat()
        }
        
        session['responses'].append(response_data)
        session['answered_questions'].add(question_id)
        session['current_question'] += 1
        
        # Update ability estimate based on IRT-like logic
        if is_correct:
            session['ability_estimate'] += 0.2
            session['consecutive_correct'] += 1
            session['consecutive_incorrect'] = 0
        else:
            session['ability_estimate'] -= 0.1
            session['consecutive_incorrect'] += 1
            session['consecutive_correct'] = 0
        
        # Calculate current performance metrics
        correct_responses = sum(1 for r in session['responses'] if r['is_correct'])
        current_score = round((correct_responses / len(session['responses'])) * 100)
        
        # Calculate knowledge level (recent performance weighted)
        if len(session['responses']) >= 3:
            recent_correct = sum(1 for r in session['responses'][-3:] if r['is_correct'])
            recent_performance = recent_correct / 3
            overall_performance = correct_responses / len(session['responses'])
            knowledge_level = 0.6 * recent_performance + 0.4 * overall_performance
        else:
            knowledge_level = correct_responses / len(session['responses'])
        
        return jsonify({
            'success': True,
            'updated_progress': {
                'ability_estimate': round(session['ability_estimate'], 2),
                'knowledge_level': round(knowledge_level, 2),
                'current_score': current_score,
                'questions_answered': len(session['responses']),
                'consecutive_correct': session['consecutive_correct'],
                'consecutive_incorrect': session['consecutive_incorrect']
            },
            'feedback': {
                'is_correct': is_correct,
                'explanation': f"{'Excellent! Keep it up!' if is_correct else 'Good effort! This helps us understand your learning needs better.'}"
            },
            'adaptation_info': {
                'current_difficulty': session['current_difficulty'],
                'next_difficulty_hint': get_student_difficulty_level(session)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to submit response: {str(e)}'}), 500

@app.route('/api/get-assessment-results', methods=['POST'])
def get_assessment_results():
    """Get final assessment results with detailed analytics"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in student_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = student_sessions[session_id]
        
        # Calculate final results
        end_time = datetime.now()
        time_spent = (end_time - session['start_time']).total_seconds() / 60
        
        correct_responses = sum(1 for r in session['responses'] if r['is_correct'])
        total_responses = len(session['responses'])
        final_score = round((correct_responses / total_responses) * 100) if total_responses > 0 else 0
        
        # Determine final ability level
        ability = session['ability_estimate']
        if ability > 1.0:
            ability_level = 'Advanced'
        elif ability > 0.0:
            ability_level = 'Intermediate'  
        else:
            ability_level = 'Beginner'
        
        # Performance by difficulty analysis
        difficulty_performance = {}
        for response in session['responses']:
            diff = response['difficulty']
            if diff not in difficulty_performance:
                difficulty_performance[diff] = {'correct': 0, 'total': 0}
            
            difficulty_performance[diff]['total'] += 1
            if response['is_correct']:
                difficulty_performance[diff]['correct'] += 1
        
        # Generate specific recommendations based on actual performance
        recommendations = generate_detailed_recommendations(session, difficulty_performance)
        
        results = {
            'success': True,
            'final_results': {
                'student_name': session['student_name'],
                'student_grade': session['student_grade'],
                'final_score': final_score,
                'ability_level': ability_level,
                'ability_estimate': round(ability, 2),
                'questions_answered': total_responses,
                'correct_answers': correct_responses,
                'time_spent_minutes': round(time_spent, 1),
                'accuracy_rate': final_score,
                'highest_difficulty_attempted': get_highest_difficulty_attempted(session),
                'adaptation_summary': get_adaptation_summary(session)
            },
            'recommendations': recommendations,
            'difficulty_performance': difficulty_performance,
            'learning_path': generate_learning_path(difficulty_performance, ability_level)
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

def generate_detailed_recommendations(session, difficulty_performance):
    """Generate detailed recommendations based on performance"""
    recommendations = []
    
    # Overall performance recommendation
    final_score = sum(1 for r in session['responses'] if r['is_correct']) / len(session['responses'])
    
    if final_score >= 0.8:
        recommendations.append({
            'icon': '🚀',
            'title': 'Excellent Performance!',
            'description': 'You\'re ready for advanced topics and challenging problems. Consider exploring competition-level mathematics.',
            'priority': 'low'
        })
    elif final_score >= 0.6:
        recommendations.append({
            'icon': '🎯',
            'title': 'Good Progress',
            'description': 'You\'re doing well! Focus on consistent practice to master moderate difficulty problems.',
            'priority': 'medium'
        })
    else:
        recommendations.append({
            'icon': '📚',
            'title': 'Build Strong Foundations',
            'description': 'Focus on mastering basic concepts before moving to complex problems. Regular practice is key.',
            'priority': 'high'
        })
    
    # Difficulty-specific recommendations
    for difficulty, performance in difficulty_performance.items():
        accuracy = performance['correct'] / performance['total']
        
        if accuracy < 0.5:
            recommendations.append({
                'icon': '💡',
                'title': f'Strengthen {difficulty} Level',
                'description': f'You answered {performance['correct']}/{performance['total']} {difficulty.lower()} questions correctly. More practice needed in this area.',
                'priority': 'high'
            })
        elif accuracy == 1.0:
            recommendations.append({
                'icon': '⭐',
                'title': f'Mastered {difficulty} Level',
                'description': f'Perfect score on {difficulty.lower()} questions! You\'re ready for the next level.',
                'priority': 'low'
            })
    
    # Consistency recommendations
    consecutive_patterns = analyze_response_patterns(session['responses'])
    if consecutive_patterns['max_incorrect_streak'] >= 3:
        recommendations.append({
            'icon': '🔄',
            'title': 'Focus on Consistency',
            'description': 'Work on maintaining focus throughout the assessment. Take breaks when needed.',
            'priority': 'medium'
        })
    
    return recommendations

def get_highest_difficulty_attempted(session):
    """Get the highest difficulty level attempted"""
    difficulties = [r['difficulty'] for r in session['responses']]
    difficulty_order = ['Very easy', 'Easy', 'Moderate', 'Difficult']
    
    for diff in reversed(difficulty_order):
        if diff in difficulties:
            return diff
    
    return 'Very easy'

def get_adaptation_summary(session):
    """Get summary of how the system adapted to the student"""
    responses = session['responses']
    if len(responses) < 2:
        return "Assessment too short for adaptation analysis"
    
    difficulties = [r['difficulty'] for r in responses]
    
    # Count difficulty changes
    changes = 0
    for i in range(1, len(difficulties)):
        if difficulties[i] != difficulties[i-1]:
            changes += 1
    
    return f"System made {changes} difficulty adjustments based on your performance"

def analyze_response_patterns(responses):
    """Analyze patterns in student responses"""
    if not responses:
        return {'max_correct_streak': 0, 'max_incorrect_streak': 0}
    
    max_correct = current_correct = 0
    max_incorrect = current_incorrect = 0
    
    for response in responses:
        if response['is_correct']:
            current_correct += 1
            current_incorrect = 0
            max_correct = max(max_correct, current_correct)
        else:
            current_incorrect += 1
            current_correct = 0
            max_incorrect = max(max_incorrect, current_incorrect)
    
    return {
        'max_correct_streak': max_correct,
        'max_incorrect_streak': max_incorrect
    }

def generate_learning_path(difficulty_performance, ability_level):
    """Generate a personalized learning path"""
    path = []
    
    if ability_level == 'Beginner':
        path = [
            "Start with Very Easy percentage problems",
            "Master basic percentage calculations", 
            "Practice Easy level word problems",
            "Build confidence with consistent practice"
        ]
    elif ability_level == 'Intermediate':
        path = [
            "Review any weak areas in Easy problems",
            "Focus on Moderate difficulty questions",
            "Practice application-based problems",
            "Prepare for advanced concepts"
        ]
    else:  # Advanced
        path = [
            "Challenge yourself with Difficult problems",
            "Explore advanced percentage applications",
            "Practice time-based problem solving",
            "Consider competitive exam preparation"
        ]
    
    return path

@app.route('/api/get-questions-preview', methods=['GET'])
def get_questions_preview():
    """Get a preview of available questions"""
    try:
        if questions_db:
            preview = {}
            for difficulty, questions in questions_db.items():
                preview[difficulty] = questions[:3]  # First 3 questions of each difficulty
            
            total_questions = sum(len(questions) for questions in questions_db.values())
            
            return jsonify({
                'success': True,
                'total_questions': total_questions,
                'questions_by_difficulty': {k: len(v) for k, v in questions_db.items()},
                'preview': preview
            })
        else:
            return jsonify({'error': 'Questions not loaded'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to get questions preview: {str(e)}'}), 500

if __name__ == '__main__':
    print("🎓 Starting AdaptIQ Adaptive Assessment System")
    print("=" * 60)
    
    # Load questions from actual dataset
    if load_questions_from_csv():
        print("✅ Questions database loaded successfully!")
        
        total_questions = sum(len(questions) for questions in questions_db.values())
        print(f"📊 Total questions available: {total_questions}")
        print("📈 Adaptive Logic: Questions will increase/decrease difficulty based on correctness")
        print("=" * 60)
        print("🌐 Server starting on http://localhost:5000")
        print("📋 Available endpoints:")
        print("  - GET  /api/health")
        print("  - POST /api/start-assessment")
        print("  - POST /api/get-question")
        print("  - POST /api/submit-response") 
        print("  - POST /api/get-assessment-results")
        print("  - GET  /api/get-questions-preview")
        print("=" * 60)
        
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load questions. Server cannot start.")