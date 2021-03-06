import argparse
from PIL import Image
import numpy as np
import onnxruntime as rt
import os
import time


def normalize_array(arr):
    normed = arr - np.min(arr)
    M = np.max(normed)
    if M == 0:
        return normed
    return normed / M

def normalize_batch(arr):
    # vectorized version of normalize_array
    # use all axes except the first one which is batch
    axis = tuple(range(1, len(arr.shape)))
    normed = arr - np.min(arr, axis=axis, keepdims=True)
    M = np.max(arr, axis=axis, keepdims=True)
    # avoid division by zero
    M[M==0] = 1
    return normed / M

def soft_sigmoid(x, a=1):
    # when a=1 this becomes usual sigmoid
    # use something like a=0.5 to stretch sigmoid
    return 1/(1 + np.exp(-a*x))

def chunked(arr, chunk_size):
  cur = 0
  while cur < len(arr):
    yield arr[cur:cur+chunk_size]
    cur += chunk_size

def array_to_image(arr):
    r = (arr *255).astype(np.uint8)
    return Image.fromarray(r, "L")

class SubjectDetector():

    def __init__(self, model_path,
                    normalize_output=False,
                    apply_soft_sigmoid=False,
                    soft_sigmoid_param=0.5):
        self.model_path = model_path
        self.normalize_output = normalize_output
        self.apply_soft_sigmoid = apply_soft_sigmoid
        self.soft_sigmoid_param = soft_sigmoid_param
        self.sess = rt.InferenceSession(self.model_path)

    def _preprocess(self, img_pil):

        img = img_pil.copy()
        img.thumbnail((256,256))
        W, H = img.size
        #print("W",W, " H",H)
        PW, PH = (256 - W) // 2, (256 - H) // 2
        #print(PW,PH)
        # HWC (RGB)
        #print(np.array(img).shape)
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        image[PH:PH+H, PW:PW+W] = np.array(img)

        image = image.astype(np.float32) / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = ((image - mean) / std).astype(np.float32)

        # CHW (RGB)
        return W, H, np.expand_dims(np.transpose(image, [2, 0, 1]), axis =0)

    def _preprocess_batch(self, images_pil):
        ims = [self._preprocess(img) for img in images_pil]
        W = [img[0] for img in ims]
        H = [img[1] for img in ims]
        tensor = np.stack([np.squeeze(img[2]) for img in ims])
        return W, H, tensor

    def _postprocess(self, result, W, H):
        if result is None:
            return None
        PW, PH = (256 - W) // 2, (256 - H) // 2
        # remove channel dimension
        r = np.squeeze(result)
        r = r[PH:PH+H, PW:PW+W]
        if self.apply_soft_sigmoid:
            r = soft_sigmoid(r, a=self.soft_sigmoid_param)
        if self.normalize_output:
            r = normalize_array(r)
        return r

    def _postprocess_batch(self, results, W, H):
        if results is None:
            return None
        # batch dimension already removed in cycle
        return [self._postprocess(result, w, h) for w, h, result in zip(W, H, results)]

    def _inference(self, x):
        # print("Input tensor shape:", x.shape)
        try:
            result = self.sess.run(None, {"input": x})
            # print("Inference session ended")
            return result[0]
        except Exception as e:
            print(str(e))
            return None

    def detect(self, list_of_images):
        w, h, tensor = self._preprocess_batch(list_of_images)
        results = self._inference(tensor)
        if results is None:
            return None
        return self._postprocess_batch(results, w, h)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for inference of ONNX model.")
    parser.add_argument("model_path", help="Path to ONNX model", type=str)
    parser.add_argument("path", help="Single image path or directory path")
    parser.add_argument("output", help="Output folder")
    parser.add_argument("--save_originals", help='Save resized originals in output', action="store_true")
    parser.add_argument("--postfix", default="", help="Postfix for output file", type=str)
    parser.add_argument('--normalize_mask', action="store_true", help='Normalize the output mask')
    parser.add_argument('--soft_sigmoid', action="store_true", help='Use soft sigmoid. Use only if the model was converted without final activation')
    
    # parser.add_argument('--use_gpu', default=False, help='Whether to use GPU or not', type=bool)

    return parser.parse_args()

if __name__ == '__main__':
    _s = time.time()
    args = parse_arguments()

    folder = os.path.dirname(args.path)

    if os.path.isfile(args.path):
        files = [os.path.basename(args.path)]
    else:
        root, dirs, files = next(os.walk(folder))

    print(f"Number of files to process: {len(files)}")

    detector = SubjectDetector(args.model_path,
            normalize_output=args.normalize_mask,
            apply_soft_sigmoid=args.soft_sigmoid,
            soft_sigmoid_param=0.5)

    for batch in chunked(files, 10):
        filenames = []
        images_pil = []
        for filename in batch:
            try:
                img = Image.open(os.path.join(folder, filename)).convert("RGB")
            except Exception as e:
                print(f"Can not open file {filename} as image, ignore.")
                print(e)
                continue
            filenames.append(filename)
            images_pil.append(img)

        if len(images_pil) == 0:
            continue
        masks = detector.detect(images_pil)
        # save masks
        for m, filename, image_pil in zip(masks, filenames, images_pil):
            img = array_to_image(m)
            new_filename = filename + "_mask_" + str(args.postfix) + ".png"
            new_path = os.path.join(args.output, new_filename)
            img.save(new_path)

            if args.save_originals:
                image_pil.thumbnail((256, 256))
                new_filename = filename + "__original" + ".png"
                new_path = os.path.join(args.output, new_filename)
                image_pil.save(new_path)
    print("All done in %.2f seconds"%(time.time() - _s))