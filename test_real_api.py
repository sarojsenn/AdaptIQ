import requests
import json

# Test the API endpoints
API_BASE_URL = 'http://localhost:5000/api'

def test_api():
    print("🧪 Testing AdaptIQ Real Dataset API")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing Health Endpoint...")
    try:
        response = requests.get(f'{API_BASE_URL}/health')
        health_data = response.json()
        print(f"✅ Status: {health_data['status']}")
        print(f"📊 Total Questions: {health_data['total_questions']}")
        print(f"📈 Questions by Difficulty: {health_data['questions_by_difficulty']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    print("\n2. Testing Start Assessment...")
    try:
        response = requests.post(f'{API_BASE_URL}/start-assessment', 
                               json={
                                   'studentName': 'Test Student',
                                   'studentGrade': '10'
                               })
        start_data = response.json()
        if start_data['success']:
            session_id = start_data['session_id']
            print(f"✅ Assessment started: {session_id}")
            print(f"🎯 Starting difficulty: {start_data['starting_difficulty']}")
        else:
            print(f"❌ Failed to start: {start_data}")
            return
    except Exception as e:
        print(f"❌ Start assessment failed: {e}")
        return
    
    print("\n3. Testing Get Question...")
    try:
        response = requests.post(f'{API_BASE_URL}/get-question',
                               json={'session_id': session_id})
        question_data = response.json()
        if question_data['success']:
            question = question_data['question']
            progress = question_data['student_progress']
            
            print(f"✅ Got question ID: {question['id']}")
            print(f"📝 Question: {question['question_text'][:50]}...")
            print(f"⚡ Difficulty: {question['difficulty']}")
            print(f"🎲 Options: A) {question['option_a'][:30]}...")
            print(f"           B) {question['option_b'][:30]}...")
            print(f"           C) {question['option_c'][:30]}...")
            print(f"           D) {question['option_d'][:30]}...")
            print(f"📊 Progress: {progress}")
        else:
            print(f"❌ Failed to get question: {question_data}")
            return
    except Exception as e:
        print(f"❌ Get question failed: {e}")
        return
    
    print("\n4. Testing Submit Response (Correct Answer)...")
    try:
        # Submit correct answer
        response = requests.post(f'{API_BASE_URL}/submit-response',
                               json={
                                   'session_id': session_id,
                                   'question_id': question['id'],
                                   'selected_option': question['answer'],
                                   'is_correct': True
                               })
        submit_data = response.json()
        if submit_data['success']:
            print(f"✅ Response submitted successfully")
            print(f"📈 Updated Progress: {submit_data['updated_progress']}")
            print(f"🎯 Adaptation Info: {submit_data['adaptation_info']}")
            print(f"💬 Feedback: {submit_data['feedback']}")
        else:
            print(f"❌ Failed to submit: {submit_data}")
    except Exception as e:
        print(f"❌ Submit response failed: {e}")
    
    print("\n5. Testing Questions Preview...")
    try:
        response = requests.get(f'{API_BASE_URL}/get-questions-preview')
        preview_data = response.json()
        if preview_data['success']:
            print(f"✅ Preview data received")
            print(f"📊 Total Questions: {preview_data['total_questions']}")
            print(f"📈 Distribution: {preview_data['questions_by_difficulty']}")
            
            # Show sample questions from each difficulty
            for difficulty, samples in preview_data['preview'].items():
                print(f"\n🎯 {difficulty} Level Sample:")
                for i, sample in enumerate(samples[:2]):  # Show 2 samples
                    print(f"   {i+1}. {sample['question_text'][:60]}...")
        else:
            print(f"❌ Preview failed: {preview_data}")
    except Exception as e:
        print(f"❌ Preview failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API Test Complete!")
    print("🌐 Open the assessment page: http://localhost:8000")
    print("📁 Assessment file: RealDatasetAssessment.html")

if __name__ == "__main__":
    test_api()