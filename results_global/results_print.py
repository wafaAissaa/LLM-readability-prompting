import os
import csv
import re


def parse_results_to_csv(output_csv="aggregated_results.csv"):
    txt_files = [f for f in os.listdir('.') if f.startswith("results") and f.endswith(".txt")]

    rows = []

    for file in txt_files:
        with open(file, 'r') as f:
            lines = f.readlines()

        model_result = [file]  # Start with the filename

        for line in lines:
            match = re.match(r".*?\s+([0-9.]+)\s±\s([0-9.]+)", line.strip())
            if match:
                mean, std = match.groups()
                mean = round(float(mean) * 100, 2)
                std = round(float(std) * 100, 2)
                model_result.append(f"{mean} ± {std}")

        if len(model_result) == 4:  # 1 name + 3 metrics
            rows.append(model_result)
        else:
            print(f"Warning: Skipping {file} due to missing metrics")

    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "accuracy", "adjacent_accuracy", "macro_f1"])
        writer.writerows(rows)

    print(f"Saved results to {output_csv}")

parse_results_to_csv()