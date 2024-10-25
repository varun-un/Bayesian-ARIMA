import json
import os

class MetadataManager:
    """
    Class to manage metadata for the data pipeline.
    """

    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """
        Load metadata from JSON file.
        """
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as file:
                metadata = json.load(file)
            return metadata
        else:
            return {}
    
    def save_metadata(self):
        """
        Save metadata to JSON file.
        """
        with open(self.metadata_path, 'w') as file:
            json.dump(self.metadata, file, indent=4)
    
    def update_metadata(self, key, value):
        """
        Update metadata with new key-value pair.
        """
        self.metadata[key] = value
        self.save_metadata()
    
    def get_metadata(self, key):
        """
        Retrieve metadata value for key.
        """
        return self.metadata[key]
    
    def get_sector(self, ticker):
        """
        Get sector for a given stock ticker.
        """
        return self.metadata[ticker]["industry"]
    
    def get_size(self, ticker):
        """
        Get company/cap size for a given stock ticker.
        """
        return self.metadata[ticker]["cap"]
    
    def get_tickers(self):
        """
        Get list of stock tickers we have metadata for.
        """
        return list(self.metadata.keys())
    

if __name__ == "__main__":
    metadata_path = "data/metadata/metadata.json"
    manager = MetadataManager(metadata_path)
    print(manager.get_metadata("AAPL"))
    print(manager.get_sector("AAPL"))
    print(manager.get_size("AAPL"))
    print(manager.get_tickers())

    
