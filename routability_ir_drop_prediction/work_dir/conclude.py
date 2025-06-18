import os
import re
import csv
import math

def extract_and_summarize_benchmarks(
    directory="/data/home/qyjh/ML4IC/CircuitNet/routability_ir_drop_prediction/work_dir/ptq_all", 
):
    output_detailed_csv=os.path.join(directory, "benchmarks_detailed.csv")
    output_summary_csv=os.path.join(directory, "benchmarks_summary.csv")
    """
    Scans a directory for files like 'component_bits.txt', extracts benchmark
    data, saves it to a detailed CSV, and also calculates and saves
    summary statistics to another CSV file.

    Args:
        directory (str): The path to the directory containing the result files.
        output_detailed_csv (str): The name for the CSV with detailed results.
        output_summary_csv (str): The name for the CSV with summary statistics.
    """
    # Regex to find the benchmark lines and extract their values.
    metric_patterns = {
        "NRMS": re.compile(r"===> Avg\. NRMS: ([\d.]+)"),
        "SSIM": re.compile(r"===> Avg\. SSIM: ([\d.]+)"),
        "EMD": re.compile(r"===> Avg\. EMD: ([\d.]+)")
    }

    # This list will store the data from each file before writing to CSV.
    all_results = []

    # Regex to identify result files and extract the component and bit number.
    # e.g., matches 'decoder_16bits.txt' and extracts 'decoder' and '16'.
    file_pattern = re.compile(r"(\w+)_(\d+)bits\.txt")

    print(f"Scanning directory '{os.path.abspath(directory)}'...")

    # Iterate over every file in the specified directory.
    for filename in os.listdir(directory):
        match = file_pattern.match(filename)
        # Proceed only if the filename matches our pattern.
        if match:
            component, bits = match.groups()
            filepath = os.path.join(directory, filename)
            
            file_data = {"Component": component, "Bits": int(bits)}
            
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                    found_all_metrics = True
                    for key, pattern in metric_patterns.items():
                        search_result = pattern.search(content)
                        if search_result:
                            file_data[key] = float(search_result.group(1))
                        else:
                            print(f"Warning: Metric '{key}' not found in '{filename}'.")
                            found_all_metrics = False
                
                if found_all_metrics:
                    all_results.append(file_data)
                else:
                    print(f"Skipping file '{filename}' due to missing metrics.")

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

    if not all_results:
        print("No valid result files found. No CSV files will be created.")
        return

    # --- Write Detailed CSV ---
    try:
        # Sort results for a clean output file.
        all_results.sort(key=lambda x: (x['Component'], x['Bits']))
        
        with open(output_detailed_csv, 'w', newline='') as f:
            fieldnames = ["Component", "Bits", "NRMS", "SSIM", "EMD"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nSuccessfully created '{output_detailed_csv}' with {len(all_results)} entries.")

    except Exception as e:
        print(f"Error writing to detailed CSV file: {e}")

    # --- Calculate Statistics and Write Summary CSV ---
    summary_data = []
    metrics_to_summarize = ["NRMS", "SSIM", "EMD"]

    for metric in metrics_to_summarize:
        values = [res[metric] for res in all_results if metric in res]
        if not values:
            continue

        # Calculate statistics
        count = len(values)
        mean = sum(values) / count
        minimum = min(values)
        maximum = max(values)
        
        # Calculate standard deviation
        variance = sum([(x - mean) ** 2 for x in values]) / count
        std_dev = math.sqrt(variance)

        summary_data.append({
            "Metric": metric,
            "Mean": mean,
            "Standard Deviation": std_dev,
            "Min": minimum,
            "Max": maximum
        })

    if not summary_data:
        print("Could not calculate summary statistics.")
        return
        
    try:
        with open(output_summary_csv, 'w', newline='') as f:
            fieldnames = ["Metric", "Mean", "Standard Deviation", "Min", "Max"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
            
        print(f"Successfully created summary file '{output_summary_csv}'.")

    except Exception as e:
        print(f"Error writing to summary CSV file: {e}")


# To run the script, save it and execute it from the command line.
# Make sure it is in the same directory as your '.txt' files.
if __name__ == "__main__":
    extract_and_summarize_benchmarks()
