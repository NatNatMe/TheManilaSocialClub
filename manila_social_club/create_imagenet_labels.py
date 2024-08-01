import urllib.request

# URL to the ImageNet class labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Download the labels
response = urllib.request.urlopen(labels_url)
labels = response.read().decode('utf-8').split("\n")

# Save the labels to a file
with open("imagenet_labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

print("imagenet_labels.txt file has been created.")
