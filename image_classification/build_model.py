import tensorflow as tf
import tflite_model_maker
from tflite_model_maker import configs
from tflite_model_maker import ImageClassifierDataLoader

# Make sure we are using Tensorflow 2
assert tf.__version__.startswith('2')

# Load the image data
data = ImageClassifierDataLoader.from_folder('images')
train_data, test_data = data.split(0.9)

# Build the model
config = configs.QuantizationConfig.create_full_integer_quantization(representative_data=test_data, is_integer_only=True)
model = tflite_model_maker.image_classifier.create(train_data)

# Evaluate the model
loss, accuracy = model.evaluate(test_data)
print("loss", loss, "accuracy", accuracy)

# Export the model
#model.export(export_dir='model', tflite_filename='model.tflite', quantization_config=config, with_metadata=False)
model.export(export_dir='model', tflite_filename='model.tflite', quantization_config=config)
