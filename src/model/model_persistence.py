import pickle

def save_model(model, filepath: str):
    """
    Serialize and save the model to a file.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str):
    """
    Load and deserialize the model from a file.
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
