import json
import csv
import re

def convert_logs_to_csv(log_file_path, csv_output_path):
    # For activities log (semicolon-delimited)
    activities_entries = []
    
    # For sandbox logs (JSON)
    sandbox_entries = []
    
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Try to parse as JSON (sandbox logs)
            try:
                log_entry = json.loads(line)
                # Extract the timestamp and lambdaLog content
                timestamp = log_entry.get('timestamp', '')
                lambda_log = log_entry.get('lambdaLog', '')
                
                # Parse the lambda log to extract trade information
                for log_line in lambda_log.split('\n'):
                    if 'BUY' in log_line or 'SELL' in log_line:
                        parts = log_line.split(': ')
                        if len(parts) >= 2:
                            product = parts[0]
                            trade_info = parts[1].split(' @ ')
                            if len(trade_info) >= 2:
                                action = trade_info[0].split(' ')[0]  # BUY or SELL
                                quantity = trade_info[0].split(' ')[1]  # Quantity
                                price = trade_info[1]  # Price
                                sandbox_entries.append({
                                    'timestamp': timestamp,
                                    'product': product,
                                    'action': action,
                                    'quantity': quantity,
                                    'price': price
                                })
                
                # If there are no trades parsed but we have a timestamp, record it
                if not any(entry['timestamp'] == timestamp for entry in sandbox_entries) and timestamp:
                    sandbox_entries.append({
                        'timestamp': timestamp,
                        'product': '',
                        'action': '',
                        'quantity': '',
                        'price': ''
                    })
                
            except json.JSONDecodeError:
                # Not JSON, try to parse as activities log format
                if ';' in line:
                    parts = line.split(';')
                    # Check if it matches the activities log format
                    if len(parts) > 3 and parts[0].isdigit() and parts[2] in ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']:
                        activities_entries.append(parts)
    
    # Determine which format had more entries
    if len(activities_entries) > len(sandbox_entries):
        # Write activities log format to CSV
        with open(csv_output_path, 'w', newline='') as csvfile:
            # Get headers from the first line if it looks like a header
            if activities_entries and not activities_entries[0][0].isdigit():
                headers = activities_entries[0]
                data_entries = activities_entries[1:]
            else:
                # Create default headers based on the structure we observed
                headers = ['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 
                           'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3',
                           'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2',
                           'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss']
                data_entries = activities_entries
            
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for entry in data_entries:
                # Ensure we have the right number of columns
                while len(entry) < len(headers):
                    entry.append('')
                writer.writerow(entry)
        
        print(f"Activities log format detected. CSV saved to {csv_output_path}")
    
    else:
        # Write sandbox logs to CSV
        with open(csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'product', 'action', 'quantity', 'price']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in sandbox_entries:
                writer.writerow(entry)
        
        print(f"Sandbox log format detected. CSV saved to {csv_output_path}")

# Example usage
convert_logs_to_csv('r1.log', 'r1converted.csv')