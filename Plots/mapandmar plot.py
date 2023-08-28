import re
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

# Ensure the directory for saving SVG files exists
output_dir = "plots_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Read the log file
with open('/home/eflinspy/Music/log.log', 'r') as f:
    log_lines = f.readlines()

class_map = {
    "chair": "Pig Sitting",
    "sofa": "Pig Standing",
    "table": "Pig Laterally",
    "bed": "Pig Sterally"
}

metrics = {class_name: {
    'AP_0.25': [],
    'AR_0.25': [],
    'AP_0.50': [],
    'AR_0.50': []
} for class_name in class_map.keys()}

eval_epochs = []

# Multiple patterns for extraction for each class
patterns = {class_name: [
    re.compile(
        rf"Epoch\(val\) \[(\d+)\].+{class_name}\s*AP_0.25:\s*(\S+),?.*?{class_name}_rec_0.25:\s*(\S+),?.*?{class_name}_AP_0.50:\s*(\S+),?.*?{class_name}_rec_0.50:\s*(\S+)"),
    re.compile(
        rf"Epoch\(val\) \[(\d+)\].+{class_name}_AP_0.25: (\S+),?.*?{class_name}_rec_0.25: (\S+),?.*?{class_name}_AP_0.50: (\S+),?.*?{class_name}_rec_0.50: (\S+)"),
    re.compile(
        rf"{class_name}_AP_0.25: (\S+),?.*?{class_name}_rec_0.25: (\S+),?.*?{class_name}_AP_0.50: (\S+),?.*?{class_name}_rec_0.50: (\S+)")
] for class_name in class_map.keys()}

for line in log_lines:
    for class_name, patterns_list in patterns.items():
        for pattern in patterns_list:
            match = pattern.search(line)
            if match:
                if len(match.groups()) == 5:
                    epoch, AP_0_25, AR_0_25, AP_0_50, AR_0_50 = match.groups()
                    if int(epoch) not in eval_epochs:
                        eval_epochs.append(int(epoch))
                else:
                    AP_0_25, AR_0_25, AP_0_50, AR_0_50 = match.groups()

                metrics[class_name]['AP_0.25'].append(float(AP_0_25.strip(',')))
                metrics[class_name]['AR_0.25'].append(float(AR_0_25.strip(',')))
                metrics[class_name]['AP_0.50'].append(float(AP_0_50.strip(',')))
                metrics[class_name]['AR_0.50'].append(float(AR_0_50.strip(',')))

                break  # Break out of the inner loop once a match is found

# Plotting for each class
for class_name, data in metrics.items():
    plt.figure(figsize=(12, 6))

    # Plotting for each class on a single graph
    for class_name, data in metrics.items():
        # Plotting Precision-Recall curve for DIoU threshold 0.50
        plt.plot(data['AR_0.50'], data['AP_0.50'], label=f"{class_map[class_name]} @ 0.50 DIoU", marker='o')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for Hardset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(f"{output_dir}/combined_PR_curve.svg", format="svg")
    plt.show()