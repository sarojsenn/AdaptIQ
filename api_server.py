from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for the model
model = None
student_sessions = {}

def load_trained_model():
    """Load the trained adaptive assessment model"""
    global model
    print("🚀 Loading adaptive assessment model...")
    
    # For this demonstration, we'll use the demo model
    # In production, you would load the actual trained IRT/BKT model
    model = create_demo_model()
    print("✅ Demo model loaded successfully!")
    return True

def create_demo_model():
    """Create a demo model with sample questions"""
    demo_model = {
        'questions': [
            {
                'id': 1,
                'question_text': 'What is 25% of 80?',
                'option_a': '20',
                'option_b': '25',
                'option_c': '15',
                'option_d': '30',
                'answer': 'a',
                'difficulty': 'Easy'
            },
            {
                'id': 2,
                'question_text': 'If x + 15 = 30, what is x?',
                'option_a': '10',
                'option_b': '15',
                'option_c': '20',
                'option_d': '45',
                'answer': 'b',
                'difficulty': 'Easy'
            },
            {
                'id': 3,
                'question_text': 'Calculate 45 × 12',
                'option_a': '540',
                'option_b': '530',
                'option_c': '520',
                'option_d': '550',
                'answer': 'a',
                'difficulty': 'Moderate'
            },
            {
                'id': 4,
                'question_text': 'What is the square root of 144?',
                'option_a': '11',
                'option_b': '12',
                'option_c': '13',
                'option_d': '14',
                'answer': 'b',
                'difficulty': 'Moderate'
            },
            {
                'id': 5,
                'question_text': 'Solve: 3x + 7 = 22',
                'option_a': '3',
                'option_b': '4',
                'option_c': '5',
                'option_d': '6',
                'answer': 'c',
                'difficulty': 'Moderate'
            },
            {
                'id': 6,
                'question_text': 'If 2^x = 32, what is x?',
                'option_a': '4',
                'option_b': '5',
                'option_c': '6',
                'option_d': '7',
                'answer': 'b',
                'difficulty': 'Difficult'
            },
            {
                'id': 7,
                'question_text': 'What is 15% of 240?',
                'option_a': '30',
                'option_b': '32',
                'option_c': '36',
                'option_d': '40',
                'answer': 'c',
                'difficulty': 'Easy'
            },
            {
                'id': 8,
                'question_text': 'Calculate: 7² + 3²',
                'option_a': '56',
                'option_b': '58',
                'option_c': '60',
                'option_d': '62',
                'answer': 'b',
                'difficulty': 'Moderate'
            },
            {
                'id': 9,
                'question_text': 'If y = 3x + 2 and x = 4, what is y?',
                'option_a': '12',
                'option_b': '14',
                'option_c': '16',
                'option_d': '18',
                'answer': 'b',
                'difficulty': 'Moderate'
            },
            {
                'id': 10,
                'question_text': 'What is 120 ÷ 8?',
                'option_a': '14',
                'option_b': '15',
                'option_c': '16',
                'option_d': '17',
                'answer': 'b',
                'difficulty': 'Easy'
            }
        ],
        'student_abilities': {},
        'student_knowledge': {}
    }
    
    print("✅ Demo model created with 10 sample questions")
    return demo_model

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'total_questions': len(model.get('questions', [])) if isinstance(model, dict) else len(model.questions_df) if hasattr(model, 'questions_df') and model.questions_df is not None else 0
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Model not loaded'
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
        student_id = f"student_{session_id}"
        
        # Initialize session
        student_sessions[session_id] = {
            'student_id': student_id,
            'student_name': student_name,
            'student_grade': student_grade,
            'start_time': datetime.now(),
            'current_question': 0,
            'responses': [],
            'answered_questions': set(),
            'ability': 0.0,
            'knowledge_level': 0.1
        }
        
        # Initialize student in model
        if model:
            if isinstance(model, dict):
                model['student_abilities'][student_id] = 0.0
                model['student_knowledge'][student_id] = {}
            else:
                model.irt_model.student_abilities[student_id] = 0.0
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'Assessment session started for {student_name}'
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
        student_id = session['student_id']
        
        # Get student progress to determine target difficulty
        if model:
            # Simplified progress calculation for demo model
            if isinstance(model, dict):
                ability = model['student_abilities'].get(student_id, 0.0)
                
                # Calculate knowledge based on previous responses
                correct_responses = sum(1 for r in session['responses'] if r['is_correct'])
                total_responses = len(session['responses'])
                knowledge = correct_responses / total_responses if total_responses > 0 else 0.1
                
                # Update session with latest progress
                session['ability'] = ability
                session['knowledge_level'] = knowledge
                
                # Determine target difficulty based on performance
                if knowledge < 0.25:
                    target_difficulty = 'Easy'
                elif knowledge < 0.5:
                    target_difficulty = 'Easy'
                elif knowledge < 0.75:
                    target_difficulty = 'Moderate'
                else:
                    target_difficulty = 'Difficult'
                
                # Select next question
                available_questions = [q for q in model['questions'] 
                                     if q['id'] not in session['answered_questions']]
                
                if not available_questions:
                    return jsonify({
                        'success': False,
                        'message': 'No more questions available',
                        'assessment_complete': True
                    })
                
                # Try to find question matching target difficulty
                target_questions = [q for q in available_questions if q['difficulty'] == target_difficulty]
                next_question = target_questions[0] if target_questions else available_questions[0]
                
                # Predict expected performance (simplified)
                difficulty_probs = {'Easy': 0.8, 'Moderate': 0.6, 'Difficult': 0.4}
                predicted_prob = difficulty_probs.get(next_question['difficulty'], 0.5)
                predicted_prob += (ability * 0.2)  # Adjust based on ability
                predicted_prob = max(0.1, min(0.9, predicted_prob))
                
                return jsonify({
                    'success': True,
                    'question': next_question,
                    'student_progress': {
                        'ability': round(ability, 2),
                        'knowledge_level': round(knowledge, 2),
                        'target_difficulty': target_difficulty,
                        'predicted_success_probability': round(predicted_prob, 2)
                    }
                })
            else:
                # Full model implementation
                progress = model.get_student_progress(student_id)
                ability = progress['ability']
                knowledge = progress['overall_knowledge']
                
                # Update session with latest progress
                session['ability'] = ability
                session['knowledge_level'] = knowledge
                
                # Determine target difficulty based on performance
                if knowledge < 0.25:
                    target_difficulty = 'Very easy'
                elif knowledge < 0.5:
                    target_difficulty = 'Easy'
                elif knowledge < 0.75:
                    target_difficulty = 'Moderate'
                else:
                    target_difficulty = 'Difficult'
                
                # Select next question using the model
                next_question = model.select_next_question(
                    student_id, 
                    session['answered_questions'], 
                    target_difficulty
                )
                
                if next_question is None:
                    return jsonify({
                        'success': False,
                        'message': 'No more questions available',
                        'assessment_complete': True
                    })
                
                # Convert to dict and prepare response
                question_data = {
                    'id': int(next_question['id']),
                    'question_text': str(next_question.get('question_text', '')),
                    'option_a': str(next_question.get('option_a', '')),
                    'option_b': str(next_question.get('option_b', '')),
                    'option_c': str(next_question.get('option_c', '')),
                    'option_d': str(next_question.get('option_d', '')),
                    'difficulty': str(next_question.get('difficulty', 'Moderate'))
                }
                
                # Predict expected performance
                predicted_prob = model.predict_performance(student_id, next_question['id'])
                
                return jsonify({
                    'success': True,
                    'question': question_data,
                    'student_progress': {
                        'ability': round(ability, 2),
                        'knowledge_level': round(knowledge, 2),
                        'target_difficulty': target_difficulty,
                        'predicted_success_probability': round(predicted_prob, 2)
                    }
                })
        else:
            return jsonify({'error': 'Model not available'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to get question: {str(e)}'}), 500

@app.route('/api/submit-response', methods=['POST'])
def submit_response():
    """Submit a student's response and update the model"""
    try:
        data = request.json
        session_id = data.get('session_id')
        question_id = data.get('question_id')
        selected_option = data.get('selected_option')
        correct_option = data.get('correct_option')
        is_correct = data.get('is_correct')
        
        if session_id not in student_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = student_sessions[session_id]
        student_id = session['student_id']
        
        # Record response
        response_data = {
            'question_id': question_id,
            'selected_option': selected_option,
            'correct_option': correct_option,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        session['responses'].append(response_data)
        session['answered_questions'].add(question_id)
        session['current_question'] += 1
        
        # Update the model with the response
        if model:
            if isinstance(model, dict):
                # Update ability based on response
                current_ability = model['student_abilities'].get(student_id, 0.0)
                if is_correct:
                    model['student_abilities'][student_id] = current_ability + 0.1
                else:
                    model['student_abilities'][student_id] = current_ability - 0.05
                
                # Update knowledge tracking
                if student_id not in model['student_knowledge']:
                    model['student_knowledge'][student_id] = {}
                
                model['student_knowledge'][student_id][question_id] = is_correct
                
                # Calculate updated progress
                ability = model['student_abilities'][student_id]
                correct_responses = sum(1 for r in session['responses'] if r['is_correct'])
                current_score = round((correct_responses / len(session['responses'])) * 100)
                knowledge_level = correct_responses / len(session['responses'])
                
                return jsonify({
                    'success': True,
                    'updated_progress': {
                        'ability': round(ability, 2),
                        'knowledge_level': round(knowledge_level, 2),
                        'current_score': current_score,
                        'questions_answered': len(session['responses'])
                    },
                    'feedback': {
                        'is_correct': is_correct,
                        'explanation': get_explanation(question_id, is_correct)
                    }
                })
            else:
                # Full model implementation
                model.update_student_model(student_id, question_id, is_correct)
                
                # Get updated progress
                progress = model.get_student_progress(student_id)
                
                # Calculate current score
                correct_responses = sum(1 for r in session['responses'] if r['is_correct'])
                current_score = round((correct_responses / len(session['responses'])) * 100)
                
                return jsonify({
                    'success': True,
                    'updated_progress': {
                        'ability': round(progress['ability'], 2),
                        'knowledge_level': round(progress['overall_knowledge'], 2),
                        'current_score': current_score,
                        'questions_answered': len(session['responses'])
                    },
                    'feedback': {
                        'is_correct': is_correct,
                        'explanation': get_explanation(question_id, is_correct)
                    }
                })
        else:
            return jsonify({'error': 'Model not available'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to submit response: {str(e)}'}), 500

@app.route('/api/get-assessment-results', methods=['POST'])
def get_assessment_results():
    """Get final assessment results and recommendations"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in student_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = student_sessions[session_id]
        student_id = session['student_id']
        
        # Calculate final results
        end_time = datetime.now()
        time_spent = (end_time - session['start_time']).total_seconds() / 60  # minutes
        
        correct_responses = sum(1 for r in session['responses'] if r['is_correct'])
        total_responses = len(session['responses'])
        final_score = round((correct_responses / total_responses) * 100) if total_responses > 0 else 0
        
        # Get final progress from model
        final_progress = None
        ability = 0
        knowledge_level = 0.1
        
        if model:
            if isinstance(model, dict):
                ability = model['student_abilities'].get(student_id, 0.0)
                knowledge_level = correct_responses / total_responses if total_responses > 0 else 0.1
                final_progress = {
                    'ability': ability,
                    'overall_knowledge': knowledge_level
                }
            else:
                final_progress = model.get_student_progress(student_id)
                ability = final_progress['ability']
                knowledge_level = final_progress['overall_knowledge']
        
        # Determine ability level
        if ability > 1.0:
            ability_level = 'Advanced'
        elif ability > 0.0:
            ability_level = 'Intermediate'
        else:
            ability_level = 'Beginner'
        
        # Generate recommendations
        recommendations = generate_recommendations(session, final_progress)
        
        # Performance by difficulty
        difficulty_performance = {}
        for response in session['responses']:
            # Get question difficulty (simplified - in production would lookup from model)
            difficulty = 'Moderate'  # Default
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = {'correct': 0, 'total': 0}
            
            difficulty_performance[difficulty]['total'] += 1
            if response['is_correct']:
                difficulty_performance[difficulty]['correct'] += 1
        
        results = {
            'success': True,
            'final_results': {
                'student_name': session['student_name'],
                'student_grade': session['student_grade'],
                'final_score': final_score,
                'ability_level': ability_level,
                'ability_score': round(ability, 2),
                'knowledge_level': round(knowledge_level, 2),
                'questions_answered': total_responses,
                'correct_answers': correct_responses,
                'time_spent_minutes': round(time_spent, 1),
                'accuracy_rate': final_score
            },
            'recommendations': recommendations,
            'difficulty_performance': difficulty_performance,
            'session_summary': {
                'total_adaptations': calculate_adaptations(session),
                'response_pattern': analyze_response_pattern(session['responses'])
            }
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

def get_explanation(question_id, is_correct):
    """Get explanation for a question (simplified)"""
    if is_correct:
        return "Excellent! You got it right."
    else:
        return "Not quite right, but good effort! Review this topic area."

def generate_recommendations(session, progress):
    """Generate personalized learning recommendations"""
    recommendations = []
    
    # Calculate performance metrics
    correct_responses = sum(1 for r in session['responses'] if r['is_correct'])
    total_responses = len(session['responses'])
    accuracy = correct_responses / total_responses if total_responses > 0 else 0
    
    # Performance-based recommendations
    if accuracy < 0.5:
        recommendations.append({
            'icon': '📚',
            'title': 'Focus on Fundamentals',
            'description': 'Review basic concepts and practice foundational problems to build a strong base.',
            'priority': 'high'
        })
    elif accuracy < 0.75:
        recommendations.append({
            'icon': '🎯',
            'title': 'Targeted Practice',
            'description': 'Focus on moderate difficulty problems to build confidence and skills.',
            'priority': 'medium'
        })
    else:
        recommendations.append({
            'icon': '🚀',
            'title': 'Advanced Challenges',
            'description': 'You\'re ready for more challenging problems and advanced topics.',
            'priority': 'low'
        })
    
    # Knowledge level recommendations
    if progress and progress['overall_knowledge'] < 0.6:
        recommendations.append({
            'icon': '💡',
            'title': 'Concept Reinforcement',
            'description': 'Spend more time on understanding core concepts before moving to complex problems.',
            'priority': 'medium'
        })
    
    # Add general study tips
    recommendations.append({
        'icon': '⏰',
        'title': 'Regular Practice',
        'description': 'Practice consistently for 15-20 minutes daily to maintain and improve your skills.',
        'priority': 'low'
    })
    
    return recommendations

def calculate_adaptations(session):
    """Calculate how many times the system adapted to the student"""
    # Simplified calculation - in production would track actual adaptations
    return len(session['responses']) // 3  # Rough estimate

def analyze_response_pattern(responses):
    """Analyze the pattern of student responses"""
    if not responses:
        return 'No responses recorded'
    
    # Calculate streak information
    current_streak = 0
    max_streak = 0
    streak_type = None
    
    for response in reversed(responses):  # Most recent first
        if current_streak == 0:
            current_streak = 1
            streak_type = 'correct' if response['is_correct'] else 'incorrect'
        elif (response['is_correct'] and streak_type == 'correct') or \
             (not response['is_correct'] and streak_type == 'incorrect'):
            current_streak += 1
        else:
            break
    
    max_streak = current_streak
    
    # Determine pattern
    if max_streak >= 3:
        if streak_type == 'correct':
            return f'Strong performance - {max_streak} correct answers in a row'
        else:
            return f'Struggling - {max_streak} incorrect answers in a row'
    else:
        return 'Mixed performance pattern'

@app.route('/api/get-questions-preview', methods=['GET'])
def get_questions_preview():
    """Get a preview of available questions for testing"""
    try:
        if model:
            if isinstance(model, dict):
                questions = model.get('questions', [])
                preview_questions = questions[:5]  # First 5 questions
                
                # Calculate difficulty distribution
                difficulty_dist = {}
                for q in questions:
                    diff = q.get('difficulty', 'Moderate')
                    difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
                
                return jsonify({
                    'success': True,
                    'total_questions': len(questions),
                    'preview_questions': preview_questions,
                    'difficulty_distribution': difficulty_dist
                })
            elif hasattr(model, 'questions_df') and model.questions_df is not None:
                # Get first 5 questions as preview
                preview_questions = []
                for idx, question in model.questions_df.head(5).iterrows():
                    preview_questions.append({
                        'id': int(question['id']),
                        'question_text': str(question.get('question_text', 'Sample question')),
                        'difficulty': str(question.get('difficulty', 'Moderate')),
                        'tags': str(question.get('tags', ''))
                    })
                
                return jsonify({
                    'success': True,
                    'total_questions': len(model.questions_df),
                    'preview_questions': preview_questions,
                    'difficulty_distribution': model.questions_df['difficulty'].value_counts().to_dict()
                })
        
        return jsonify({'error': 'No questions available'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to get questions preview: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting AdaptIQ Assessment API Server...")
    print("=" * 50)
    
    # Load the trained model
    if load_trained_model():
        print("✅ Model loaded successfully!")
        total_questions = 0
        if model:
            if isinstance(model, dict):
                total_questions = len(model.get('questions', []))
            elif hasattr(model, 'questions_df') and model.questions_df is not None:
                total_questions = len(model.questions_df)
        
        print(f"📊 Total questions available: {total_questions}")
        print("=" * 50)
        print("🌐 Server starting on http://localhost:5000")
        print("📋 Available endpoints:")
        print("  - GET  /api/health")
        print("  - POST /api/start-assessment")
        print("  - POST /api/get-question")
        print("  - POST /api/submit-response")
        print("  - POST /api/get-assessment-results")
        print("  - GET  /api/get-questions-preview")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Server cannot start.")