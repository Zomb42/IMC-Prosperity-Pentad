import json
import csv
import re # Import regular expressions

def parse_sandbox_log_trades(log_file_path, csv_output_path):
    """
    Parses sandbox-style JSON logs to extract trade execution information using regex.
    Writes the extracted trades to a CSV file.
    """
    trade_entries = []

    # Regex to capture BUYING/SELLING actions, quantity, product, and price
    # Handles variations in spacing reasonably well.
    # Groups: 1=Action, 2=Quantity, 3=Product, 4=Price
    trade_pattern = re.compile(
        r"(BUYING|SELLING)\s+(\d+)\s+([A-Z_0-9]+)\s+@\s+(\d+(\.\d+)?)"
        #r"^\s*(BUYING|SELLING)\s+(\d+)\s+([A-Z_0-9]+)\s+@\s+(\d+(\.\d+)?)\s*$" # More strict line match
    )

    print(f"Processing log file: {log_file_path}")
    line_count = 0
    json_errors = 0
    trades_found = 0

    with open(log_file_path, 'r') as file:
        for i, line in enumerate(file):
            line_count += 1
            line = line.strip()
            if not line:
                continue # Skip empty lines

            try:
                log_entry = json.loads(line)
                timestamp = log_entry.get('timestamp')
                lambda_log = log_entry.get('lambdaLog', '')

                if not timestamp or not lambda_log:
                    continue # Skip if essential fields are missing

                # Find all trade matches within the lambdaLog content
                matches = trade_pattern.finditer(lambda_log)

                found_trade_in_line = False
                for match in matches:
                    action = match.group(1)
                    quantity = match.group(2)
                    product = match.group(3)
                    price = match.group(4)

                    trade_entries.append({
                        'timestamp': timestamp,
                        'product': product,
                        'action': action,
                        'quantity': quantity,
                        'price': price
                    })
                    trades_found += 1
                    found_trade_in_line = True
                
                # Optional: Log if a JSON line had no trades (for debugging)
                # if not found_trade_in_line:
                #    print(f"Debug: No trades found in lambdaLog at timestamp {timestamp}")


            except json.JSONDecodeError:
                json_errors += 1
                # Optionally print lines that failed JSON parsing
                # print(f"Warning: Line {i+1} is not valid JSON: {line[:100]}...")
                continue # Skip non-JSON lines silently
            except Exception as e:
                 print(f"Error processing line {i+1}: {e}")


    print(f"Processed {line_count} lines.")
    print(f"Found {trades_found} trades.")
    if json_errors > 0:
        print(f"Encountered {json_errors} lines that were not valid JSON.")

    # Write the extracted trades to CSV
    if not trade_entries:
        print("No trades found to write to CSV.")
        return

    with open(csv_output_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'product', 'action', 'quantity', 'price']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(trade_entries)

    print(f"Successfully wrote {len(trade_entries)} trades to {csv_output_path}")


# --- Example Usage ---
# Replace 'r3try1.log' with the actual path to your Round 3 log file
log_file = 'r3try1.log'
csv_file = 'r3_trades_parsed.csv' # Use a new name to avoid confusion

parse_sandbox_log_trades(log_file, csv_file)