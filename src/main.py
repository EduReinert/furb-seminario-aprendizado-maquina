import subprocess
import time

def run_models():
    """Execute both Random Forest and SVM models"""
    print("Starting protein expression analysis...\n")
    
    # Run Random Forest
    print("\n" + "="*50)
    print("RUNNING RANDOM FOREST CLASSIFIER")
    print("="*50)
    start_time = time.time()
    subprocess.run(["python", "random_forest.py"])
    print(f"\nRandom Forest completed in {time.time() - start_time:.2f} seconds")
    
    # Run SVM
    print("\n" + "="*50)
    print("RUNNING SVM CLASSIFIER")
    print("="*50)
    start_time = time.time()
    subprocess.run(["python", "svm.py"])
    print(f"\nSVM completed in {time.time() - start_time:.2f} seconds")
    
    print("\nAnalysis complete! Check the generated plots:")

if __name__ == "__main__":
    run_models()