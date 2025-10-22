"""
Test script to verify the complete AQI prediction pipeline
Run this to ensure all components are working correctly
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import sklearn
        import joblib
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def test_data_generation():
    """Test data generation script"""
    print("\nTesting data generation...")
    try:
        import generate_sample_data
        if os.path.exists('data/air_quality_india.csv'):
            print("✓ Data file created successfully")
            return True
        else:
            print("✗ Data file not found")
            return False
    except Exception as e:
        print(f"✗ Data generation error: {e}")
        return False

def test_model_creation():
    """Test if model can be created and trained"""
    print("\nTesting model creation...")
    try:
        from aqi_predictor import AQIPredictor
        
        predictor = AQIPredictor(model_type='random_forest')
        print("✓ Model object created successfully")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_model_training():
    """Test model training"""
    print("\nTesting model training...")
    try:
        from aqi_predictor import AQIPredictor
        
        if not os.path.exists('data/air_quality_india.csv'):
            print("✗ Data file not found. Run generate_sample_data.py first")
            return False
        
        predictor = AQIPredictor(model_type='random_forest')
        metrics = predictor.train('data/air_quality_india.csv', test_size=0.2)
        
        # Check if metrics are reasonable
        if metrics['test_r2'] > 0.8:
            print(f"✓ Model trained successfully (R² = {metrics['test_r2']:.4f})")
            return True
        else:
            print(f"✗ Model performance below threshold (R² = {metrics['test_r2']:.4f})")
            return False
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False

def test_predictions():
    """Test prediction functionality"""
    print("\nTesting predictions...")
    try:
        from aqi_predictor import AQIPredictor
        
        predictor = AQIPredictor()
        
        # Check if model exists
        if not os.path.exists('models/aqi_model.pkl'):
            print("  Model not found, training new model...")
            predictor.train('data/air_quality_india.csv')
            predictor.save_model('models/aqi_model.pkl')
        else:
            predictor.load_model('models/aqi_model.pkl')
        
        # Load data and make prediction
        df = predictor.load_data('data/air_quality_india.csv')
        predictions = predictor.predict_future(df, 'Delhi', days_ahead=7)
        
        if len(predictions) == 7:
            print("✓ Predictions generated successfully")
            print(f"  Sample: AQI for Delhi = {predictions['Predicted_AQI'].iloc[0]:.1f}")
            return True
        else:
            print("✗ Incorrect number of predictions")
            return False
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return False

def test_deployment():
    """Test deployment script"""
    print("\nTesting deployment script...")
    try:
        import deploy_model
        print("✓ Deployment script loads successfully")
        return True
    except Exception as e:
        print(f"✗ Deployment script error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("AQI Prediction Pipeline - Test Suite")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Model Creation", test_model_creation),
        ("Model Training", test_model_training),
        ("Predictions", test_predictions),
        ("Deployment Script", test_deployment)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run: python aqi_predictor.py (to train on full dataset)")
        print("2. Run: python deploy_model.py (to make predictions)")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
