import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import json
warnings.filterwarnings('ignore')

class IRTModel:
    """
    Item Response Theory (IRT) Model using 2-Parameter Logistic Model (2PL)
    """
    def __init__(self):
        self.item_params = {}  # Dictionary to store discrimination (a) and difficulty (b) parameters
        self.student_abilities = {}  # Dictionary to store student abilities (theta)
        
    def probability(self, theta, a, b):
        """
        Calculate probability of correct response using 2PL model
        P(X=1|theta) = 1 / (1 + exp(-a(theta - b)))
        """
        try:
            return 1 / (1 + np.exp(-a * (theta - b)))
        except OverflowError:
            return 0.0 if theta < b else 1.0
    
    def fit(self, responses_df, questions_df):
        """
        Fit IRT model parameters using Maximum Likelihood Estimation
        """
        print("Fitting IRT Model...")
        
        # Initialize parameters
        n_items = len(questions_df)
        n_students = 100  # Default number of students for synthetic data
        
        # Convert difficulty levels to numeric values
        difficulty_map = {'Very easy': -1.5, 'Easy': -0.5, 'Moderate': 0.5, 'Difficult': 1.5}
        
        # Initialize item parameters based on difficulty
        for idx, row in questions_df.iterrows():
            item_id = row['id']
            difficulty_level = row['difficulty']
            
            # Initialize discrimination (a) parameter - higher for better discriminating items
            a_init = np.random.normal(1.0, 0.2)
            
            # Initialize difficulty (b) parameter based on difficulty level
            b_init = difficulty_map.get(difficulty_level, 0.0) + np.random.normal(0, 0.1)
            
            self.item_params[item_id] = {'a': max(0.1, a_init), 'b': b_init}
        
        # Initialize student abilities (theta)
        for student_id in range(n_students):
            self.student_abilities[f'student_{student_id}'] = np.random.normal(0, 1)
        
        print(f"Initialized parameters for {n_items} items and {n_students} students")
        return self
    
    def predict_probability(self, student_id, item_id):
        """
        Predict probability of correct response for a student-item pair
        """
        if student_id not in self.student_abilities or item_id not in self.item_params:
            return 0.5  # Default probability
        
        theta = self.student_abilities[student_id]
        a = self.item_params[item_id]['a']
        b = self.item_params[item_id]['b']
        
        return self.probability(theta, a, b)
    
    def get_item_difficulty(self, item_id):
        """
        Get item difficulty parameter
        """
        return self.item_params.get(item_id, {}).get('b', 0)
    
    def get_item_discrimination(self, item_id):
        """
        Get item discrimination parameter
        """
        return self.item_params.get(item_id, {}).get('a', 1)


class BKTModel:
    """
    Bayesian Knowledge Tracing Model
    """
    def __init__(self):
        # BKT Parameters
        self.prior_knowledge = 0.1  # P(L0) - Initial knowledge probability
        self.learning_rate = 0.2    # P(T) - Probability of learning from instruction
        self.slip_probability = 0.1  # P(S) - Probability of making a mistake when knowing
        self.guess_probability = 0.2 # P(G) - Probability of guessing correctly when not knowing
        
        # Student knowledge states
        self.student_knowledge = {}
        
    def update_knowledge(self, student_id, item_id, response_correct, item_difficulty):
        """
        Update student's knowledge state based on response using BKT
        """
        if student_id not in self.student_knowledge:
            self.student_knowledge[student_id] = {}
        
        if item_id not in self.student_knowledge[student_id]:
            # Initialize with prior knowledge, adjusted by item difficulty
            difficulty_adjustment = max(0.05, 1.0 - abs(item_difficulty) * 0.2)
            self.student_knowledge[student_id][item_id] = self.prior_knowledge * difficulty_adjustment
        
        current_knowledge = self.student_knowledge[student_id][item_id]
        
        if response_correct:
            # Student got it right
            # P(L_n+1 | correct) = (P(L_n) * (1 - P(S))) / (P(L_n) * (1 - P(S)) + (1 - P(L_n)) * P(G))
            numerator = current_knowledge * (1 - self.slip_probability)
            denominator = numerator + (1 - current_knowledge) * self.guess_probability
        else:
            # Student got it wrong
            # P(L_n+1 | incorrect) = (P(L_n) * P(S)) / (P(L_n) * P(S) + (1 - P(L_n)) * (1 - P(G)))
            numerator = current_knowledge * self.slip_probability
            denominator = numerator + (1 - current_knowledge) * (1 - self.guess_probability)
        
        if denominator > 0:
            updated_knowledge = numerator / denominator
        else:
            updated_knowledge = current_knowledge
        
        # Apply learning rate
        final_knowledge = updated_knowledge + (1 - updated_knowledge) * self.learning_rate
        
        # Ensure knowledge stays within [0, 1]
        self.student_knowledge[student_id][item_id] = max(0.01, min(0.99, final_knowledge))
        
        return self.student_knowledge[student_id][item_id]
    
    def get_knowledge_state(self, student_id, item_id):
        """
        Get current knowledge state for student-item pair
        """
        if student_id not in self.student_knowledge:
            return self.prior_knowledge
        return self.student_knowledge[student_id].get(item_id, self.prior_knowledge)
    
    def predict_performance(self, student_id, item_id):
        """
        Predict probability of correct response based on knowledge state
        """
        knowledge = self.get_knowledge_state(student_id, item_id)
        # P(correct) = P(L) * (1 - P(S)) + (1 - P(L)) * P(G)
        return knowledge * (1 - self.slip_probability) + (1 - knowledge) * self.guess_probability


class AdaptiveAssessmentModel:
    """
    Combined IRT + BKT Model for Adaptive Assessment
    """
    def __init__(self):
        self.irt_model = IRTModel()
        self.bkt_model = BKTModel()
        self.questions_df = None
        self.difficulty_levels = ['Very easy', 'Easy', 'Moderate', 'Difficult']
        
    def load_data_safely(self, questions_path, responses_path):
        """
        Load questions and responses data with error handling
        """
        print("Loading data...")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    print(f"Trying encoding: {encoding}")
                    self.questions_df = pd.read_csv(questions_path, encoding=encoding)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.questions_df is None:
                raise Exception("Could not read the CSV file with any encoding")
            
            # Clean the data
            self.questions_df = self.questions_df.dropna(subset=['id'])
            self.questions_df = self.questions_df[self.questions_df['id'].apply(lambda x: str(x).isdigit())]
            self.questions_df['id'] = self.questions_df['id'].astype(int)
            
            # Fill missing difficulty with 'Moderate'
            self.questions_df['difficulty'] = self.questions_df['difficulty'].fillna('Moderate')
            
            print(f"Loaded {len(self.questions_df)} questions")
            print(f"Difficulty distribution:")
            print(self.questions_df['difficulty'].value_counts())
            
            return self.questions_df, None
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating synthetic dataset for training...")
            return self.create_synthetic_dataset()
    
    def create_synthetic_dataset(self):
        """
        Create a synthetic dataset for training when real data is unavailable
        """
        print("Creating synthetic question dataset...")
        
        # Create synthetic questions
        questions_data = []
        question_templates = [
            "What is {a} + {b}?",
            "Calculate {a} × {b}",
            "What is {a}% of {b}?",
            "If x + {a} = {b}, what is x?",
            "Simplify: {a}/{b}",
            "What is the square root of {a}?",
            "Convert {a} to percentage",
            "Find {a}% of {b}",
            "Solve: {a}x = {b}",
            "What is {a} - {b}?"
        ]
        
        difficulties = ['Very easy', 'Easy', 'Moderate', 'Difficult']
        
        for i in range(1, 81):  # Create 80 questions
            template = np.random.choice(question_templates)
            
            # Generate numbers based on difficulty
            if i <= 20:  # Very easy
                a, b = np.random.randint(1, 10, 2)
                difficulty = 'Very easy'
            elif i <= 40:  # Easy
                a, b = np.random.randint(5, 50, 2)
                difficulty = 'Easy'
            elif i <= 60:  # Moderate
                a, b = np.random.randint(10, 100, 2)
                difficulty = 'Moderate'
            else:  # Difficult
                a, b = np.random.randint(50, 500, 2)
                difficulty = 'Difficult'
            
            question_text = template.format(a=a, b=b)
            
            questions_data.append({
                'id': i,
                'question_text': question_text,
                'option_a': f"{a+b}",
                'option_b': f"{a*2}",
                'option_c': f"{b*2}",
                'option_d': f"{a+b+1}",
                'answer': 'a',  # Simplified for demo
                'difficulty': difficulty,
                'tags': 'Mathematics'
            })
        
        self.questions_df = pd.DataFrame(questions_data)
        
        print(f"Created {len(self.questions_df)} synthetic questions")
        print("Difficulty distribution:")
        print(self.questions_df['difficulty'].value_counts())
        
        return self.questions_df, None
    
    def create_synthetic_student_data(self, n_students=200, n_responses_per_student=25):
        """
        Create synthetic student response data for training
        """
        print(f"Creating synthetic training data for {n_students} students...")
        
        synthetic_data = []
        
        for student_id in range(n_students):
            # Assign student ability (normally distributed)
            student_ability = np.random.normal(0, 1)
            
            # Select random questions for this student
            selected_questions = self.questions_df.sample(n=min(n_responses_per_student, len(self.questions_df)))
            
            for _, question in selected_questions.iterrows():
                item_id = question['id']
                difficulty = question['difficulty']
                
                # Convert difficulty to numeric
                difficulty_map = {'Very easy': -1.5, 'Easy': -0.5, 'Moderate': 0.5, 'Difficult': 1.5}
                item_difficulty = difficulty_map.get(difficulty, 0)
                
                # Simulate response based on student ability and item difficulty
                prob_correct = 1 / (1 + np.exp(-(student_ability - item_difficulty)))
                
                # Add some noise
                prob_correct = max(0.1, min(0.9, prob_correct + np.random.normal(0, 0.1)))
                
                response_correct = np.random.random() < prob_correct
                
                synthetic_data.append({
                    'student_id': f'student_{student_id}',
                    'item_id': item_id,
                    'response_correct': response_correct,
                    'difficulty': difficulty,
                    'student_ability': student_ability,
                    'item_difficulty': item_difficulty
                })
        
        return pd.DataFrame(synthetic_data)
    
    def train(self, n_students=200):
        """
        Train the combined IRT + BKT model
        """
        print("Training Adaptive Assessment Model...")
        
        # Create synthetic training data
        training_data = self.create_synthetic_student_data(n_students)
        
        # Train IRT model
        self.irt_model.fit(training_data, self.questions_df)
        
        # Train BKT model by processing responses sequentially
        print("Training BKT model...")
        
        # Group by student and process responses chronologically
        for student_id, student_data in training_data.groupby('student_id'):
            for _, response in student_data.iterrows():
                item_id = response['item_id']
                response_correct = response['response_correct']
                item_difficulty = response['item_difficulty']
                
                # Update knowledge state
                self.bkt_model.update_knowledge(student_id, item_id, response_correct, item_difficulty)
        
        print("Training completed!")
        return self
    
    def select_next_question(self, student_id, answered_questions=None, target_difficulty='Moderate'):
        """
        Select the next optimal question for a student using adaptive logic
        """
        if answered_questions is None:
            answered_questions = set()
        
        # Get available questions (not yet answered)
        available_questions = self.questions_df[~self.questions_df['id'].isin(answered_questions)]
        
        if len(available_questions) == 0:
            return None
        
        best_question = None
        best_score = -float('inf')
        
        for _, question in available_questions.iterrows():
            item_id = question['id']
            
            # Get IRT prediction
            irt_prob = self.irt_model.predict_probability(student_id, item_id)
            
            # Get BKT prediction
            bkt_prob = self.bkt_model.predict_performance(student_id, item_id)
            
            # Combined probability
            combined_prob = (irt_prob + bkt_prob) / 2
            
            # Calculate information value (closer to 0.5 is more informative)
            information = 1 - abs(combined_prob - 0.5) * 2
            
            # Difficulty matching bonus
            difficulty_bonus = 1.0 if question['difficulty'] == target_difficulty else 0.5
            
            # Final score
            score = information * difficulty_bonus
            
            if score > best_score:
                best_score = score
                best_question = question
        
        return best_question
    
    def update_student_model(self, student_id, item_id, response_correct):
        """
        Update both IRT and BKT models based on student response
        """
        item_difficulty = self.irt_model.get_item_difficulty(item_id)
        
        # Update BKT knowledge state
        self.bkt_model.update_knowledge(student_id, item_id, response_correct, item_difficulty)
        
        # For IRT, adjust student ability based on response
        if student_id not in self.irt_model.student_abilities:
            self.irt_model.student_abilities[student_id] = 0.0
        
        current_ability = self.irt_model.student_abilities[student_id]
        if response_correct:
            self.irt_model.student_abilities[student_id] = current_ability + 0.1
        else:
            self.irt_model.student_abilities[student_id] = current_ability - 0.1
    
    def predict_performance(self, student_id, item_id):
        """
        Predict student performance on an item using combined model
        """
        irt_prob = self.irt_model.predict_probability(student_id, item_id)
        bkt_prob = self.bkt_model.predict_performance(student_id, item_id)
        
        # Weighted combination (can be tuned)
        combined_prob = 0.6 * irt_prob + 0.4 * bkt_prob
        
        return combined_prob
    
    def get_student_progress(self, student_id):
        """
        Get comprehensive student progress report
        """
        if student_id not in self.bkt_model.student_knowledge:
            return {'overall_knowledge': 0.1, 'item_knowledge': {}, 'ability': 0.0}
        
        item_knowledge = self.bkt_model.student_knowledge[student_id]
        overall_knowledge = np.mean(list(item_knowledge.values())) if item_knowledge else 0.1
        
        ability = self.irt_model.student_abilities.get(student_id, 0.0)
        
        return {
            'overall_knowledge': overall_knowledge,
            'item_knowledge': item_knowledge,
            'ability': ability,
            'num_items_attempted': len(item_knowledge)
        }
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        model_data = {
            'irt_model': self.irt_model,
            'bkt_model': self.bkt_model,
            'questions_df': self.questions_df
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        self.irt_model = model_data['irt_model']
        self.bkt_model = model_data['bkt_model']
        self.questions_df = model_data['questions_df']
        print(f"Model loaded from {filepath}")
    
    def to_json(self):
        """
        Convert model to JSON for web interface
        """
        questions_list = []
        for _, question in self.questions_df.iterrows():
            questions_list.append({
                'id': int(question['id']),
                'question_text': str(question.get('question_text', '')),
                'option_a': str(question.get('option_a', '')),
                'option_b': str(question.get('option_b', '')),
                'option_c': str(question.get('option_c', '')),
                'option_d': str(question.get('option_d', '')),
                'answer': str(question.get('answer', 'a')),
                'difficulty': str(question.get('difficulty', 'Moderate')),
                'tags': str(question.get('tags', ''))
            })
        
        return {
            'questions': questions_list,
            'model_info': {
                'total_questions': len(self.questions_df),
                'difficulties': self.questions_df['difficulty'].value_counts().to_dict()
            }
        }


def main():
    """
    Main function to train and save the adaptive assessment model
    """
    print("AdaptIQ - Training Adaptive Assessment Model")
    print("=" * 50)
    
    # Initialize model
    model = AdaptiveAssessmentModel()
    
    # Load data (will create synthetic if real data fails)
    questions_df, responses_df = model.load_data_safely(
        'data/questions.csv', 
        'data/responses.csv'
    )
    
    # Train model
    model.train(n_students=300)  # Train with more synthetic students
    
    # Save trained model
    model.save_model('trained_adaptive_model.pkl')
    
    # Save questions as JSON for web interface
    with open('questions_data.json', 'w') as f:
        json.dump(model.to_json(), f, indent=2)
    
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Files created:")
    print("- trained_adaptive_model.pkl (trained model)")
    print("- questions_data.json (questions for web interface)")
    
    # Quick demo
    print("\nQuick Demo:")
    student_id = "demo_student"
    
    for i in range(3):
        progress = model.get_student_progress(student_id)
        print(f"\nQuestion {i+1}:")
        print(f"Student ability: {progress['ability']:.2f}")
        
        next_question = model.select_next_question(student_id, set())
        if next_question is not None:
            print(f"Selected question: {next_question['question_text']}")
            print(f"Difficulty: {next_question['difficulty']}")
            
            # Simulate response
            predicted_prob = model.predict_performance(student_id, next_question['id'])
            response_correct = np.random.random() < predicted_prob
            
            model.update_student_model(student_id, next_question['id'], response_correct)
            print(f"Response: {'Correct' if response_correct else 'Incorrect'}")


if __name__ == "__main__":
    main()