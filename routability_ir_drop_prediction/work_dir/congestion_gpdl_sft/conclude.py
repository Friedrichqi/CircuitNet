import os
import re
import csv

def extract_benchmarks(directory=".", output_csv="benchmarks.csv"):
    """
    Scans a directory for files matching 'results_*.txt', extracts benchmark
    data, and compiles it into a single CSV file.

    Args:
        directory (str): The path to the directory containing the result files.
                         Defaults to the current directory.
        output_csv (str): The name of the output CSV file.
                          Defaults to 'benchmarks.csv'.
    """
    # Regex to find the benchmark lines and extract their values.
    # We capture the floating-point number at the end of the line.
    patterns = {
        "NRMS": re.compile(r"===> Avg\. NRMS: ([\d.]+)"),
        "SSIM": re.compile(r"===> Avg\. SSIM: ([\d.]+)"),
        "EMD": re.compile(r"===> Avg\. EMD: ([\d.]+)")
    }

    # This list will store the data from each file before writing to CSV.
    all_results = []

    # Regex to identify the result files and extract the iteration number.
    file_pattern = re.compile(r"results_(\d+)\.txt")

    print(f"Scanning directory '{os.path.abspath(directory)}'...")

    # Iterate over every file in the specified directory.
    for filename in os.listdir(directory):
        match = file_pattern.match(filename)
        # Proceed only if the filename matches our 'results_*.txt' pattern.
        if match:
            iteration_num = match.group(1)
            filepath = os.path.join(directory, filename)
            
            # This dictionary will hold the extracted data for the current file.
            file_data = {"Iteration": iteration_num}
            
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                    # Search for each benchmark pattern in the file content.
                    for key, pattern in patterns.items():
                        search_result = pattern.search(content)
                        if search_result:
                            # Convert the extracted string to a float and store it.
                            file_data[key] = float(search_result.group(1))
                
                # After checking the file, add data to our main list if all
                # benchmarks were found.
                if len(file_data) == len(patterns) + 1: # +1 for "Iteration"
                    all_results.append(file_data)
                else:
                    print(f"Warning: Skipped '{filename}'. Not all benchmarks were found.")

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

    # After checking all files, write the collected data to a CSV.
    if not all_results:
        print("No valid result files found. CSV file will not be created.")
        return

    # Sort results by iteration number for a clean output file.
    all_results.sort(key=lambda x: int(x['Iteration']))

    try:
        with open(output_csv, 'w', newline='') as f:
            # Define the CSV column headers.
            fieldnames = ["Iteration", "NRMS", "SSIM", "EMD"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write the header row.
            writer.writeheader()
            # Write all the data rows.
            writer.writerows(all_results)
        
        print(f"\nSuccessfully created '{output_csv}' with {len(all_results)} entries.")

    except Exception as e:
        print(f"Error writing to CSV file: {e}")


# To run the script, just execute this python file.
# Make sure it is in the same directory as your 'results_*.txt' files,
# or change the directory argument in the function call below.
if __name__ == "__main__":
    extract_benchmarks()
