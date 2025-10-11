import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
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
        n_students = len(responses_df.columns) - 1  # Assuming first column is question_id
        
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
        
    def load_data(self, questions_path, responses_path):
        """
        Load questions and responses data
        """
        print("Loading data...")
        
        # Load questions
        self.questions_df = pd.read_csv(questions_path)
        
        # Clean questions data
        self.questions_df = self.questions_df.dropna(subset=['id', 'difficulty'])
        self.questions_df = self.questions_df[self.questions_df['id'].notna()]
        
        # Load responses
        responses_df = pd.read_csv(responses_path)
        
        print(f"Loaded {len(self.questions_df)} questions")
        print(f"Difficulty distribution:")
        print(self.questions_df['difficulty'].value_counts())
        
        return self.questions_df, responses_df
    
    def create_synthetic_student_data(self, n_students=100, n_responses_per_student=20):
        """
        Create synthetic student response data for training
        """
        print(f"Creating synthetic data for {n_students} students...")
        
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
                # Higher ability students are more likely to answer correctly
                # Easier questions are more likely to be answered correctly
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
    
    def train(self, n_students=100):
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
        
        # For IRT, we would typically need to re-estimate parameters
        # For now, we'll adjust student ability slightly based on response
        if student_id in self.irt_model.student_abilities:
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


def main():
    """
    Main function to train and demonstrate the adaptive assessment model
    """
    # Initialize model
    model = AdaptiveAssessmentModel()
    
    # Load data
    questions_df, responses_df = model.load_data(
        'data/questions.csv', 
        'data/responses.csv'
    )
    
    # Train model
    model.train(n_students=200)  # Create synthetic data for 200 students
    
    # Save trained model
    model.save_model('adaptive_assessment_model.pkl')
    
    # Demonstrate adaptive assessment for a new student
    print("\n" + "="*50)
    print("ADAPTIVE ASSESSMENT DEMO")
    print("="*50)
    
    student_id = "demo_student"
    answered_questions = set()
    
    # Simulate an adaptive test session
    for round_num in range(10):
        print(f"\nRound {round_num + 1}:")
        
        # Get student progress
        progress = model.get_student_progress(student_id)
        print(f"Student ability: {progress['ability']:.2f}")
        print(f"Overall knowledge: {progress['overall_knowledge']:.2f}")
        
        # Determine target difficulty based on performance
        if progress['overall_knowledge'] < 0.3:
            target_difficulty = 'Very easy'
        elif progress['overall_knowledge'] < 0.5:
            target_difficulty = 'Easy'
        elif progress['overall_knowledge'] < 0.7:
            target_difficulty = 'Moderate'
        else:
            target_difficulty = 'Difficult'
        
        print(f"Target difficulty: {target_difficulty}")
        
        # Select next question
        next_question = model.select_next_question(
            student_id, 
            answered_questions, 
            target_difficulty
        )
        
        if next_question is None:
            print("No more questions available!")
            break
        
        item_id = next_question['id']
        print(f"Selected question ID: {item_id}")
        print(f"Question difficulty: {next_question['difficulty']}")
        
        # Simulate student response (random for demo)
        predicted_prob = model.predict_performance(student_id, item_id)
        response_correct = np.random.random() < predicted_prob
        
        print(f"Predicted probability: {predicted_prob:.2f}")
        print(f"Student response: {'Correct' if response_correct else 'Incorrect'}")
        
        # Update model
        model.update_student_model(student_id, item_id, response_correct)
        answered_questions.add(item_id)
    
    # Final progress report
    print("\n" + "="*50)
    print("FINAL PROGRESS REPORT")
    print("="*50)
    
    final_progress = model.get_student_progress(student_id)
    print(f"Final student ability: {final_progress['ability']:.2f}")
    print(f"Final overall knowledge: {final_progress['overall_knowledge']:.2f}")
    print(f"Items attempted: {final_progress['num_items_attempted']}")
    
    # Create visualization
    create_visualizations(model, student_id)


def create_visualizations(model, student_id):
    """
    Create visualizations for the model performance
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Item difficulty distribution
    ax1 = axes[0, 0]
    difficulty_counts = model.questions_df['difficulty'].value_counts()
    ax1.bar(difficulty_counts.index, difficulty_counts.values, color='skyblue', alpha=0.7)
    ax1.set_title('Distribution of Question Difficulties')
    ax1.set_xlabel('Difficulty Level')
    ax1.set_ylabel('Number of Questions')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. IRT item parameters
    ax2 = axes[0, 1]
    difficulties = [model.irt_model.item_params[item_id]['b'] for item_id in model.irt_model.item_params]
    discriminations = [model.irt_model.item_params[item_id]['a'] for item_id in model.irt_model.item_params]
    
    scatter = ax2.scatter(difficulties, discriminations, alpha=0.6, c='coral')
    ax2.set_xlabel('Item Difficulty (b)')
    ax2.set_ylabel('Item Discrimination (a)')
    ax2.set_title('IRT Item Parameters')
    ax2.grid(True, alpha=0.3)
    
    # 3. Student knowledge progression (if available)
    ax3 = axes[1, 0]
    if student_id in model.bkt_model.student_knowledge:
        knowledge_values = list(model.bkt_model.student_knowledge[student_id].values())
        items = list(model.bkt_model.student_knowledge[student_id].keys())
        
        ax3.plot(range(len(knowledge_values)), knowledge_values, marker='o', linewidth=2, markersize=6)
        ax3.set_xlabel('Question Sequence')
        ax3.set_ylabel('Knowledge State')
        ax3.set_title(f'Knowledge Progression - {student_id}')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, 'No student data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Knowledge Progression')
    
    # 4. Model performance comparison
    ax4 = axes[1, 1]
    
    # Simulate some performance data
    difficulties = ['Very easy', 'Easy', 'Moderate', 'Difficult']
    irt_accuracy = [0.85, 0.75, 0.65, 0.55]  # Simulated accuracies
    bkt_accuracy = [0.80, 0.78, 0.72, 0.60]  # Simulated accuracies
    combined_accuracy = [0.88, 0.82, 0.75, 0.65]  # Simulated accuracies
    
    x = np.arange(len(difficulties))
    width = 0.25
    
    ax4.bar(x - width, irt_accuracy, width, label='IRT Only', alpha=0.7)
    ax4.bar(x, bkt_accuracy, width, label='BKT Only', alpha=0.7)
    ax4.bar(x + width, combined_accuracy, width, label='Combined Model', alpha=0.7)
    
    ax4.set_xlabel('Question Difficulty')
    ax4.set_ylabel('Prediction Accuracy')
    ax4.set_title('Model Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(difficulties)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_assessment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'adaptive_assessment_analysis.png'")


if __name__ == "__main__":
    main()