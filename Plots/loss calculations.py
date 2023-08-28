import re
import matplotlib.pyplot as plt
import os

# Ensure the directory for saving SVG files exists
output_dir = "plots_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the log file
with open('/home/eflinspy/Music/log.log', 'r') as f:
    log_lines = f.readlines()

# Structures to store the training and evaluation metrics
metrics_data = {
    "epoch": [],
    "total_loss": [],
    "bbox_loss": [],
    "cls_loss": []
}

# Regular expressions for extraction
train_pattern = re.compile(r"Epoch \[(\d+)\].+bbox_loss: (\S+), cls_loss: (\S+), loss: (\S+),")

# Extract training metrics from the log lines
for line in log_lines:
    match = train_pattern.search(line)
    if match:
        epoch, bbox_loss, cls_loss, total_loss = match.groups()
        metrics_data["epoch"].append(int(epoch))
        metrics_data["bbox_loss"].append(float(bbox_loss))
        metrics_data["cls_loss"].append(float(cls_loss))
        metrics_data["total_loss"].append(float(total_loss))


# Plotting total loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(metrics_data["epoch"], metrics_data["total_loss"], label="Total Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Total Loss over Epochs for Hardset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/total_loss_over_epochs.svg", format="svg")
plt.close()

# Plotting bbox_loss and cls_loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(metrics_data["epoch"], metrics_data["bbox_loss"], label="Bounding Box Loss", color="red")
plt.plot(metrics_data["epoch"], metrics_data["cls_loss"], label="Classification Loss", color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Bounding Box and Classification Losses over Epochs for Hardset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/bbox_cls_loss_over_epochs.svg", format="svg")
plt.close()