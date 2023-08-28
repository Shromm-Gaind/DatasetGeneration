import matplotlib.pyplot as plt
import re

# Your log file path
log_path = '/home/eflinspy/Dataset/log2.log'

# Open the log file and read its contents
with open(log_path, 'r') as log_file:
    log = log_file.read()

# Define a pattern to extract the relevant data
pattern = r"Saving checkpoint at (\d+) epochs.*?Overall\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)"

# Find all matches of the pattern in the log
matches = re.findall(pattern, log, re.DOTALL)

# Separate the data
epochs, ap_025, ar_025, ap_050, ar_050 = zip(*matches)

# Convert to the appropriate types
epochs = [int(epoch) for epoch in epochs]
ap_025 = [float(ap) for ap in ap_025]
ar_025 = [float(ar) for ar in ar_025]
ap_050 = [float(ap) for ap in ap_050]
ar_050 = [float(ar) for ar in ar_050]

# Now we can plot the data
plt.figure(figsize=(10, 5))

plt.plot(epochs, ap_025, label='mAP_0.25 DIoU')
plt.plot(epochs, ar_025, label='mAR_0.25 DIoU')
plt.plot(epochs, ap_050, label='mAP_0.50 DIoU')
plt.plot(epochs, ar_050, label='mAR_0.50 DIoU')

plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Performance metrics over epochs for easyset')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an SVG file before showing
plt.savefig('/home/eflinspy/Music/ploteasy.png', format='png')
plt.show()
