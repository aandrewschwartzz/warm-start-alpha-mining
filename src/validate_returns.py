from data_loader import DataLoader

def main():
    # Example list of stock symbols - you can modify this list
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Create data loader instance
    loader = DataLoader(symbols)
    
    # Run validation
    loader.validate_returns()

if __name__ == "__main__":
    main() 