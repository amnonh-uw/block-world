from record_io import record_io
import numpy as np
from PIL import Image

def dump(path, type_dict):
    io = record_io(None, type_dict)
    sample = io.load_sample(path)

    for key in sample:
        tf_type = type_dict[key]
        if tf_type == "img8" or tf_type == "img16":
            print("{}: {} {}".format(key, sample[key].shape, sample[key].dtype))
            if tf_type == "img8":
                img = Image.fromarray(sample[key])
                img.save(key + ".png")
            else:
                img = sample[key]
                img = img.astype(np.uint16)
                img = Image.fromarray(img)
                img.save(key + ".tiff")
        else:
            print("{}: {} {} {}".format(key, sample[key].shape, sample[key].dtype, sample[key]))
