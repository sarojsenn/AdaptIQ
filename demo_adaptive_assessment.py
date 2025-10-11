"""
Adaptive Assessment Demo Script
This script demonstrates how the adaptive assessment system works with real student interactions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adaptive_assessment_model import AdaptiveAssessmentModel
import pandas as pd
import numpy as np

class AdaptiveAssessmentDemo:
    def __init__(self):
        self.model = AdaptiveAssessmentModel()
        self.load_and_train_model()
    
    def load_and_train_model(self):
        """Load data and train the model"""
        print("Loading and training adaptive assessment model...")
        
        # Load data
        try:
            questions_df, responses_df = self.model.load_data(
                'data/questions.csv', 
                'data/responses.csv'
            )
            
            # Train model with synthetic data
            self.model.train(n_students=150)
            
            print("Model training completed successfully!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating minimal demo with sample data...")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data if files are not accessible"""
        sample_questions = pd.DataFrame({
            'id': range(1, 21),
            'question_text': [f'Sample question {i}' for i in range(1, 21)],
            'difficulty': ['Very easy'] * 5 + ['Easy'] * 5 + ['Moderate'] * 5 + ['Difficult'] * 5,
            'answer': ['a', 'b', 'c', 'd'] * 5
        })
        
        self.model.questions_df = sample_questions
        self.model.train(n_students=50)
    
    def run_adaptive_test(self, student_id, num_questions=10):
        """Run an adaptive test for a student"""
        print(f"\n{'='*60}")
        print(f"ADAPTIVE ASSESSMENT SESSION FOR {student_id}")
        print(f"{'='*60}")
        
        answered_questions = set()
        student_responses = []
        
        for question_num in range(1, num_questions + 1):
            print(f"\n--- Question {question_num} ---")
            
            # Get current student progress
            progress = self.model.get_student_progress(student_id)
            
            # Determine target difficulty based on performance
            if progress['overall_knowledge'] < 0.25:
                target_difficulty = 'Very easy'
            elif progress['overall_knowledge'] < 0.5:
                target_difficulty = 'Easy'
            elif progress['overall_knowledge'] < 0.75:
                target_difficulty = 'Moderate'
            else:
                target_difficulty = 'Difficult'
            
            print(f"Current ability estimate: {progress['ability']:.2f}")
            print(f"Overall knowledge: {progress['overall_knowledge']:.2f}")
            print(f"Target difficulty: {target_difficulty}")
            
            # Select next question
            next_question = self.model.select_next_question(
                student_id, answered_questions, target_difficulty
            )
            
            if next_question is None:
                print("No more suitable questions available!")
                break
            
            item_id = next_question['id']
            print(f"Selected Question ID: {item_id}")
            print(f"Question Difficulty: {next_question['difficulty']}")
            
            # Display question if available
            if 'question_text' in next_question and pd.notna(next_question['question_text']):
                print(f"Question: {next_question['question_text']}")
            
            # Get model's prediction
            predicted_prob = self.model.predict_performance(student_id, item_id)
            print(f"Model's predicted success probability: {predicted_prob:.2f}")
            
            # Simulate student response based on their true ability and question difficulty
            # This simulates a real student taking the test
            student_ability = progress['ability']
            difficulty_map = {'Very easy': -1.5, 'Easy': -0.5, 'Moderate': 0.5, 'Difficult': 1.5}
            item_difficulty = difficulty_map.get(next_question['difficulty'], 0)
            
            # Calculate true probability based on IRT model
            true_prob = 1 / (1 + np.exp(-(student_ability - item_difficulty)))
            true_prob = max(0.1, min(0.9, true_prob + np.random.normal(0, 0.1)))
            
            response_correct = np.random.random() < true_prob
            
            print(f"Student response: {'✓ CORRECT' if response_correct else '✗ INCORRECT'}")
            
            # Update the model with the response
            self.model.update_student_model(student_id, item_id, response_correct)
            
            # Record response
            student_responses.append({
                'question_num': question_num,
                'item_id': item_id,
                'difficulty': next_question['difficulty'],
                'target_difficulty': target_difficulty,
                'predicted_prob': predicted_prob,
                'response_correct': response_correct,
                'ability_after': self.model.get_student_progress(student_id)['ability']
            })
            
            answered_questions.add(item_id)
        
        # Final assessment report
        self.generate_assessment_report(student_id, student_responses)
        
        return student_responses
    
    def generate_assessment_report(self, student_id, responses):
        """Generate a comprehensive assessment report"""
        print(f"\n{'='*60}")
        print(f"ASSESSMENT REPORT FOR {student_id}")
        print(f"{'='*60}")
        
        # Overall statistics
        total_questions = len(responses)
        correct_answers = sum(1 for r in responses if r['response_correct'])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        print(f"Total Questions Attempted: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Overall Accuracy: {accuracy:.1%}")
        
        # Final progress
        final_progress = self.model.get_student_progress(student_id)
        print(f"Final Ability Estimate: {final_progress['ability']:.2f}")
        print(f"Final Knowledge Level: {final_progress['overall_knowledge']:.2f}")
        
        # Performance by difficulty
        print(f"\nPerformance by Difficulty Level:")
        difficulty_performance = {}
        for response in responses:
            diff = response['difficulty']
            if diff not in difficulty_performance:
                difficulty_performance[diff] = {'correct': 0, 'total': 0}
            
            difficulty_performance[diff]['total'] += 1
            if response['response_correct']:
                difficulty_performance[diff]['correct'] += 1
        
        for difficulty, stats in difficulty_performance.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {difficulty}: {stats['correct']}/{stats['total']} ({accuracy:.1%})")
        
        # Ability progression
        print(f"\nAbility Progression:")
        abilities = [r['ability_after'] for r in responses]
        for i, ability in enumerate(abilities):
            print(f"  After Q{i+1}: {ability:.2f}")
        
        # Recommendations
        self.generate_recommendations(student_id, final_progress, difficulty_performance)
    
    def generate_recommendations(self, student_id, progress, difficulty_performance):
        """Generate personalized learning recommendations"""
        print(f"\nPersonalized Learning Recommendations:")
        
        ability = progress['ability']
        knowledge = progress['overall_knowledge']
        
        if ability < -1.0:
            print("  • Focus on fundamental concepts and very easy practice problems")
            print("  • Consider additional instructional support")
        elif ability < 0.0:
            print("  • Practice with easy to moderate difficulty problems")
            print("  • Review basic concepts before moving to advanced topics")
        elif ability < 1.0:
            print("  • Ready for moderate to difficult problems")
            print("  • Can handle most standard curriculum topics")
        else:
            print("  • Excellent performance - ready for advanced and challenging problems")
            print("  • Consider enrichment activities and advanced topics")
        
        # Specific difficulty recommendations
        for difficulty, stats in difficulty_performance.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            if accuracy < 0.5:
                print(f"  • Need more practice with {difficulty.lower()} problems")
            elif accuracy > 0.8:
                print(f"  • Strong performance on {difficulty.lower()} problems - ready to advance")
    
    def run_multiple_students_demo(self):
        """Run demo with multiple students to show adaptation"""
        print(f"\n{'='*80}")
        print("MULTI-STUDENT ADAPTIVE ASSESSMENT DEMO")
        print(f"{'='*80}")
        
        # Create students with different ability levels
        students = [
            {'id': 'low_ability_student', 'initial_ability': -1.5},
            {'id': 'average_student', 'initial_ability': 0.0},
            {'id': 'high_ability_student', 'initial_ability': 1.5}
        ]
        
        all_results = {}
        
        for student in students:
            student_id = student['id']
            
            # Initialize student ability in the model
            self.model.irt_model.student_abilities[student_id] = student['initial_ability']
            
            print(f"\nTesting {student_id} (Initial ability: {student['initial_ability']})")
            results = self.run_adaptive_test(student_id, num_questions=8)
            all_results[student_id] = results
        
        # Compare adaptation across students
        self.compare_student_adaptations(all_results)
    
    def compare_student_adaptations(self, all_results):
        """Compare how the system adapted to different students"""
        print(f"\n{'='*80}")
        print("ADAPTIVE SYSTEM COMPARISON")
        print(f"{'='*80}")
        
        for student_id, results in all_results.items():
            print(f"\n{student_id.upper()}:")
            
            difficulties_given = [r['difficulty'] for r in results]
            difficulty_counts = {}
            for diff in difficulties_given:
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            
            print("  Questions by difficulty:")
            for diff, count in difficulty_counts.items():
                percentage = count / len(results) * 100
                print(f"    {diff}: {count} questions ({percentage:.1f}%)")
            
            final_ability = results[-1]['ability_after'] if results else 0
            accuracy = sum(1 for r in results if r['response_correct']) / len(results)
            
            print(f"  Final ability: {final_ability:.2f}")
            print(f"  Overall accuracy: {accuracy:.1%}")


def main():
    """Main demo function"""
    print("AdaptIQ - Adaptive Assessment System Demo")
    print("Using IRT (Item Response Theory) + BKT (Bayesian Knowledge Tracing)")
    
    # Initialize demo
    demo = AdaptiveAssessmentDemo()
    
    # Run single student demo
    print("\n1. Single Student Adaptive Test Demo")
    demo.run_adaptive_test("demo_student_1", num_questions=10)
    
    # Run multiple students demo
    print("\n2. Multiple Students Comparison Demo")
    demo.run_multiple_students_demo()
    
    print(f"\n{'='*80}")
    print("DEMO COMPLETED")
    print("The system successfully demonstrated:")
    print("• Adaptive question selection based on student performance")
    print("• Real-time ability estimation using IRT")
    print("• Knowledge state tracking using BKT")
    print("• Personalized difficulty adjustment")
    print("• Comprehensive assessment reporting")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()