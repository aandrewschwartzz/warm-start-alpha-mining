"""
Module for converting portfolio weights to dictionary format.
"""
import csv
from datetime import datetime, date
import json
import os
from typing import Dict, Any, Union

class DateEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle date objects."""
    def default(self, obj):
        if isinstance(obj, date):
            return {
                "__date__": True,
                "year": obj.year,
                "month": obj.month,
                "day": obj.day
            }
        return super().default(obj)

def convert_weights_to_dict(csv_path: str) -> Dict[date, Dict[str, float]]:
    """
    Convert portfolio weights CSV file to a dictionary format.
    
    Args:
        csv_path: Path to the CSV file containing portfolio weights
        
    Returns:
        Dictionary with dates as keys and nested dictionaries of symbol weights
    """
    portfolio_dict = {}
    
    with open(csv_path, 'r') as file:
        # Read the header to get stock symbols
        reader = csv.reader(file)
        headers = next(reader)
        stock_symbols = headers[1:]  # Skip the date column
        
        # Process each row
        for row in reader:
            date_str = row[0]
            # Convert date string to date object
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Create dictionary for this date's weights
            weights = {}
            for symbol, weight in zip(stock_symbols, row[1:]):
                weights[symbol] = float(weight)
            
            portfolio_dict[date_obj] = weights
    
    return portfolio_dict

def format_dict_with_dates(d: Union[Dict, Any], indent: int = 0) -> str:
    """
    Custom dictionary formatter that handles date objects.
    """
    if isinstance(d, dict):
        if "__date__" in d:
            # This is our date object marker
            return f"date({d['year']}, {d['month']}, {d['day']})"
        
        items = []
        for k, v in d.items():
            if isinstance(k, date):
                key_str = f"date({k.year}, {k.month}, {k.day})"
            else:
                key_str = json.dumps(k)
            
            value_str = format_dict_with_dates(v, indent + 4)
            items.append(f"{' ' * (indent + 4)}{key_str}: {value_str}")
        
        if items:
            return "{\n" + ",\n".join(items) + "\n" + (' ' * indent) + "}"
        return "{}"
    
    return json.dumps(d)

def save_dict_to_file(portfolio_dict: Dict[date, Dict[str, float]], output_path: str) -> None:
    """
    Save portfolio weights dictionary to a text file in Python format.
    
    Args:
        portfolio_dict: Dictionary containing portfolio weights
        output_path: Path where the text file should be saved
    """
    # Format the dictionary with proper date handling
    formatted_dict = format_dict_with_dates(portfolio_dict)
    
    with open(output_path, 'w') as file:
        file.write("portfolio_weights = " + formatted_dict) 