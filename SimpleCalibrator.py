import os
import logging
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from PIL import Image

# Returns a numpy buffer of shape (num_images, 1, 28, 28)
def load_data():
    np.random.seed(42)
    num_images = 100
    image_c = 3
    image_h = 432
    image_w = 768
    # Need to scale all values to the range of [0, 1]
    data = []
    for i in range(num_images):
        random_image = np.random.rand(image_c, image_h, image_w)
        data.append(random_image)

    return np.ascontiguousarray(np.array(data).astype(np.float32).reshape(num_images, image_c, image_h, image_w))


class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file="simple_calibration.cache", batch_size=12):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_data()
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


