import pandas as pd
import re

def clean_and_merge_datasets():
    """
    Clean the new questions dataset and merge with existing questions
    """
    print("🚀 Starting dataset merge and cleaning...")
    
    # Step 1: Read existing dataset
    try:
        print("📖 Reading existing questions...")
        try:
            existing_df = pd.read_csv('data/questions.csv', encoding='utf-8')
        except:
            existing_df = pd.read_csv('data/questions.csv', encoding='latin-1')
        
        print(f"✅ Existing dataset: {len(existing_df)} questions")
        
        # Get the highest ID from existing dataset
        max_existing_id = existing_df['id'].max() if not existing_df.empty else 0
        print(f"📊 Highest existing ID: {max_existing_id}")
        
    except Exception as e:
        print(f"❌ Error reading existing dataset: {e}")
        return False
    
    # Step 2: Clean and read new questions
    try:
        print("🧹 Cleaning new questions file...")
        
        # Read the file as text first to clean it
        with open('data/newquestions.csv', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean up problematic lines
        lines = content.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                # Count commas to check if line has correct number of fields
                comma_count = line.count(',')
                
                if comma_count == 8:  # Correct number of fields
                    cleaned_lines.append(line)
                elif comma_count > 8:  # Too many commas, likely due to quotes
                    # Try to fix by removing extra commas in question text
                    parts = line.split(',')
                    if len(parts) > 9:
                        # Merge the question parts and clean special characters
                        question_parts = parts[1:-7]  # Everything except id and last 7 fields
                        fixed_question = ' '.join(question_parts).replace('"', '').replace(',', '')
                        # Replace special characters
                        fixed_question = fixed_question.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                        
                        fixed_line = f"{parts[0]},{fixed_question},{','.join(parts[-7:])}"
                        cleaned_lines.append(fixed_line)
                    else:
                        cleaned_lines.append(line)
                else:
                    print(f"⚠️  Skipping malformed line {i+1}: {line[:50]}...")
        
        # Write cleaned content to temporary file
        with open('temp_cleaned_questions.csv', 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        
        # Read the cleaned file
        new_df = pd.read_csv('temp_cleaned_questions.csv', encoding='utf-8')
        print(f"✅ New questions loaded: {len(new_df)} questions")
        
    except Exception as e:
        print(f"❌ Error processing new questions: {e}")
        return False
    
    # Step 3: Adjust IDs to avoid conflicts
    try:
        print("🔢 Adjusting question IDs...")
        
        # Shift all new question IDs to start after existing ones
        id_offset = max_existing_id
        new_df['id'] = new_df['id'] + id_offset
        
        print(f"✅ New question IDs: {new_df['id'].min()} to {new_df['id'].max()}")
        
    except Exception as e:
        print(f"❌ Error adjusting IDs: {e}")
        return False
    
    # Step 4: Combine datasets
    try:
        print("🔗 Merging datasets...")
        
        # Ensure both dataframes have the same columns
        required_cols = ['id', 'question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'answer', 'difficulty', 'tags']
        
        for col in required_cols:
            if col not in existing_df.columns:
                existing_df[col] = ''
            if col not in new_df.columns:
                new_df[col] = ''
        
        # Select only required columns in correct order
        existing_df = existing_df[required_cols]
        new_df = new_df[required_cols]
        
        # Clean special characters in text columns
        text_columns = ['question_text', 'option_a', 'option_b', 'option_c', 'option_d']
        for col in text_columns:
            if col in new_df.columns:
                new_df[col] = new_df[col].astype(str).str.replace('\u2018', "'").str.replace('\u2019', "'").str.replace('\u201c', '"').str.replace('\u201d', '"')
        
        # Combine
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        print(f"✅ Combined dataset: {len(combined_df)} questions")
        
    except Exception as e:
        print(f"❌ Error combining datasets: {e}")
        return False
    
    # Step 5: Save merged dataset
    try:
        print("💾 Saving merged dataset...")
        
        # Backup original
        existing_df.to_csv('data/questions_backup.csv', index=False, encoding='utf-8')
        
        # Save combined dataset
        combined_df.to_csv('data/questions.csv', index=False, encoding='utf-8')
        
        print(f"✅ Dataset saved! Total questions: {len(combined_df)}")
        print(f"💾 Backup created: data/questions_backup.csv")
        
        # Cleanup temp file
        import os
        if os.path.exists('temp_cleaned_questions.csv'):
            os.remove('temp_cleaned_questions.csv')
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        return False

def retrain_model():
    """
    Retrain the ML model with the updated dataset
    """
    try:
        print("\n🧠 Starting model retraining...")
        
        # Import and run training
        import train_model
        train_model.main()
        
        print("✅ Model retraining completed!")
        print("📁 Updated model files:")
        print("  - trained_adaptive_model.pkl")
        print("  - questions_data.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 AdaptIQ Dataset Merger & Model Retrainer")
    print("=" * 50)
    
    # Step 1: Merge datasets
    if clean_and_merge_datasets():
        print("\n" + "=" * 50)
        
        # Step 2: Ask about retraining
        retrain_choice = input("🔄 Retrain the ML model now? (y/n): ").lower().strip()
        
        if retrain_choice in ['y', 'yes']:
            if retrain_model():
                print("\n🎉 SUCCESS! Your model is now trained with all questions!")
                print("🚀 Ready for adaptive assessments with expanded question bank!")
            else:
                print("\n⚠️  Dataset merged but model retraining failed.")
                print("💡 You can manually retrain by running: python train_model.py")
        else:
            print("\n⏳ Dataset merged successfully!")
            print("💡 Run 'python train_model.py' when you want to retrain the model.")
    
    else:
        print("\n❌ Dataset merge failed. Please check the error messages above.")