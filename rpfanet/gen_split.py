import random

# Set the total number of images
total_images = 546

# Calculate the number of images for training, testing, and validation
num_training = int(total_images * 0.6)
num_testing = int(total_images * 0.3)
num_validation = total_images - num_training - num_testing

# Generate a list of image numbers
image_numbers = [str(i).zfill(6) for i in range(total_images)]

# Shuffle the image numbers
random.shuffle(image_numbers)

# Split the image numbers into training, testing, and validation sets
training_set = image_numbers[:num_training]
testing_set = image_numbers[num_training:num_training+num_testing]
validation_set = image_numbers[num_training+num_testing:]

# Sort the training, testing, and validation sets
training_set.sort()
testing_set.sort()
validation_set.sort()

# Write the training set to training.txt
with open('train.txt', 'w') as f:
    f.write('\n'.join(training_set))

# Write the testing set to testing.txt
with open('test.txt', 'w') as f:
    f.write('\n'.join(testing_set))

# Write the validation set to validation.txt
with open('val.txt', 'w') as f:
    f.write('\n'.join(validation_set))