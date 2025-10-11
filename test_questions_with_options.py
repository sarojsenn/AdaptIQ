import requests
import json

def test_specific_questions():
    print("🧪 Testing Specific Questions with Option Issues")
    print("=" * 60)
    
    API_BASE_URL = 'http://localhost:5000/api'
    
    try:
        # Start an assessment session
        print("1. Starting assessment session...")
        start_response = requests.post(f'{API_BASE_URL}/start-assessment', 
                                     json={
                                         'studentName': 'Test Student',
                                         'studentGrade': '10'
                                     })
        
        if start_response.status_code != 200:
            print(f"❌ Failed to start session: {start_response.status_code}")
            return
            
        session_data = start_response.json()
        if not session_data['success']:
            print(f"❌ Failed to start session: {session_data}")
            return
            
        session_id = session_data['session_id']
        print(f"✅ Session started: {session_id}")
        
        # Get multiple questions to find the problematic ones
        print("\n2. Testing questions for complete options...")
        
        questions_tested = 0
        found_target_questions = []
        
        for i in range(15):  # Test up to 15 questions
            question_response = requests.post(f'{API_BASE_URL}/get-question',
                                            json={'session_id': session_id})
            
            if question_response.status_code != 200:
                print(f"❌ Failed to get question {i+1}")
                break
                
            question_data = question_response.json()
            
            if not question_data['success']:
                if question_data.get('assessment_complete'):
                    print("✅ Assessment completed")
                    break
                else:
                    print(f"❌ Failed to get question {i+1}: {question_data}")
                    break
            
            question = question_data['question']
            questions_tested += 1
            
            # Check if this is one of our target questions
            question_text = question['question_text'].lower()
            
            if any(keyword in question_text for keyword in ['270 candidate', '3 hours', 'sulphur', '2250']):
                found_target_questions.append(question)
                print(f"\n🎯 Found target question ID {question['id']}:")
                print(f"   Question: {question['question_text'][:60]}...")
                print(f"   Difficulty: {question['difficulty']}")
                
                # Check all options
                options_present = 0
                for opt in ['a', 'b', 'c', 'd']:
                    option_text = question.get(f'option_{opt}', '')
                    if option_text and option_text.strip() and option_text.strip() != 'nan':
                        options_present += 1
                        print(f"   Option {opt.upper()}: {option_text}")
                    else:
                        print(f"   Option {opt.upper()}: [MISSING]")
                
                print(f"   Correct Answer: {question['answer'].upper()}")
                print(f"   Options Status: {options_present}/4 present")
                
                if options_present == 4:
                    print("   ✅ All options present!")
                else:
                    print(f"   ❌ Only {options_present} options present")
            
            # Submit a dummy response to continue
            requests.post(f'{API_BASE_URL}/submit-response',
                         json={
                             'session_id': session_id,
                             'question_id': question['id'],
                             'selected_option': question['answer'],
                             'is_correct': True
                         })
        
        print(f"\n📊 Summary:")
        print(f"   Questions tested: {questions_tested}")
        print(f"   Target questions found: {len(found_target_questions)}")
        
        if not found_target_questions:
            print("   ℹ️  Target questions not encountered in this sample")
            print("   💡 Try refreshing the assessment to see different questions")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the API server is running on localhost:5000")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_specific_questions()