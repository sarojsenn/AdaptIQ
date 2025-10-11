import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our function
from adaptive_api_server import generate_missing_options

def test_option_generation():
    print("🧪 Testing Option Generation Logic")
    print("=" * 50)
    
    # Test case 1: Question with completely missing options
    question1 = {
        'id': 17,
        'question_text': 'What percent of day is 3 hours?',
        'option_a': '',
        'option_b': '',
        'option_c': '',
        'option_d': '',
        'answer': 'a',
        'difficulty': 'Very easy',
        'tags': 'Percentages'
    }
    
    print("\n1. Testing completely missing options:")
    print(f"   Question: {question1['question_text']}")
    print(f"   Original options: A='{question1['option_a']}', B='{question1['option_b']}', C='{question1['option_c']}', D='{question1['option_d']}'")
    
    result1 = generate_missing_options(question1)
    print(f"   Generated options: A='{result1['option_a']}', B='{result1['option_b']}', C='{result1['option_c']}', D='{result1['option_d']}'")
    print(f"   Correct answer: {result1['answer'].upper()}")
    
    # Test case 2: Question with incomplete options (like the 270 candidates question)
    question2 = {
        'id': 12,
        'question_text': '270 candidate appeared for an examination, of which 252 passed. The pass percentage is:',
        'option_a': '80%',
        'option_b': '',
        'option_c': '',
        'option_d': '',
        'answer': 'd',
        'difficulty': 'Very easy',
        'tags': 'Percentages'
    }
    
    print("\n2. Testing incomplete options:")
    print(f"   Question: {question2['question_text']}")
    print(f"   Original options: A='{question2['option_a']}', B='{question2['option_b']}', C='{question2['option_c']}', D='{question2['option_d']}'")
    
    result2 = generate_missing_options(question2)
    print(f"   Generated options: A='{result2['option_a']}', B='{result2['option_b']}', C='{result2['option_c']}', D='{result2['option_d']}'")
    print(f"   Correct answer: {result2['answer'].upper()}")
    
    # Calculate the actual correct answer for validation
    actual_percentage = (252 / 270) * 100
    print(f"   Actual calculation: 252/270 * 100 = {actual_percentage:.2f}%")
    
    # Test case 3: Question with sulphur
    question3 = {
        'id': 13,
        'question_text': '5 out of 2250 parts of earth is sulphur. What is the percentage of sulphur in earth:',
        'option_a': '',
        'option_b': '',
        'option_c': '',
        'option_d': '',
        'answer': 'b',
        'difficulty': 'Very easy',
        'tags': 'Percentages'
    }
    
    print("\n3. Testing sulphur question:")
    print(f"   Question: {question3['question_text']}")
    print(f"   Original options: A='{question3['option_a']}', B='{question3['option_b']}', C='{question3['option_c']}', D='{question3['option_d']}'")
    
    result3 = generate_missing_options(question3)
    print(f"   Generated options: A='{result3['option_a']}', B='{result3['option_b']}', C='{result3['option_c']}', D='{result3['option_d']}'")
    print(f"   Correct answer: {result3['answer'].upper()}")
    
    # Calculate the actual correct answer
    actual_sulphur = (5 / 2250) * 100
    print(f"   Actual calculation: 5/2250 * 100 = {actual_sulphur:.3f}%")
    
    print("\n" + "=" * 50)
    print("✅ Option generation test complete!")

if __name__ == "__main__":
    test_option_generation()