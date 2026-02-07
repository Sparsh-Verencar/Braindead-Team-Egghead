# utils/inspect_pickle.py

import pickle
import pandas as pd

def inspect_pickle(file_path):
    """Inspect contents of a pickle file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {file_path}")
    print('='*60)
    
    try:
        # Try pandas read_pickle first
        obj = pd.read_pickle(file_path)
        print(f"✅ Loaded with pd.read_pickle")
        print(f"Type: {type(obj)}")
        
    except Exception as e:
        print(f"⚠️ pd.read_pickle failed: {e}")
        
        # Try custom unpickler
        try:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    try:
                        return super().find_class(module, name)
                    except (AttributeError, ModuleNotFoundError):
                        return type(name, (), {})
            
            with open(file_path, 'rb') as f:
                obj = CustomUnpickler(f).load()
            print(f"✅ Loaded with CustomUnpickler")
            print(f"Type: {type(obj)}")
        except Exception as e2:
            print(f"❌ Failed completely: {e2}")
            return None
    
    # Show contents
    if isinstance(obj, dict):
        print(f"\n📦 Dictionary with {len(obj)} keys:")
        for key in obj.keys():
            val = obj[key]
            print(f"  - {key}: {type(val).__name__}", end="")
            if isinstance(val, pd.DataFrame):
                print(f" (shape: {val.shape})")
            elif hasattr(val, 'shape'):
                print(f" (shape: {val.shape})")
            else:
                print()
    
    elif hasattr(obj, '__dict__'):
        print(f"\n📦 Object with attributes:")
        for key, val in obj.__dict__.items():
            print(f"  - {key}: {type(val).__name__}", end="")
            if isinstance(val, pd.DataFrame):
                print(f" (shape: {val.shape})")
            elif hasattr(val, 'shape'):
                print(f" (shape: {val.shape})")
            else:
                print()
    else:
        print(f"\n📦 Simple object: {type(obj)}")
    
    return obj

if __name__ == "__main__":
    import sys
    import os
    
    models_dir = "D:\Braindead-Team-Egghead\models"
    
    if len(sys.argv) > 1:
        # Inspect specific file
        inspect_pickle(sys.argv[1])
    else:
        # Inspect all .pkl files in models directory
        for file in os.listdir(models_dir):
            if file.endswith('.pkl'):
                inspect_pickle(os.path.join(models_dir, file))