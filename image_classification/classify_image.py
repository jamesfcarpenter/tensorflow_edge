import tflite_runtime.interpreter as tflite
from PIL import Image
import platform
import classify
# from tflite_support import metadata as _metadata
import time

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

# displayer = _metadata.MetadataDisplayer.with_model_file('model/model.tflite')
# content = displayer.get_metadata_json()

labels = {
    0: 'bird',
    1: 'cat',
    2: 'dog'
}

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(
    model_path='model/model.tflite',
    experimental_delegates=[
        tflite.load_delegate(EDGETPU_SHARED_LIB, {})
    ])
interpreter.allocate_tensors()

#print(interpreter.get_input_details())

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

size = classify.input_size(interpreter)

for file in ['parrot.jpg', 'dog.jpg', 'cat.jpg']:
    image = Image.open(file).convert('RGB').resize(size, Image.ANTIALIAS)
    classify.set_input(interpreter, image)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    print('Inference time: %.1fms' % (inference_time * 1000))
    classes = classify.get_output(interpreter, 1, 0.0)
    for klass in classes:
        print("file", file, "label", labels[klass.id], "score", klass.score)
