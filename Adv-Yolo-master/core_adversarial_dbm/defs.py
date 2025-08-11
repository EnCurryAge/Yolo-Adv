import pathlib

import torch as T

# ROOT_PATH = pathlib.Path(__file__).parent.parent.parent
ROOT_PATH = pathlib.Path("D:/Yolo-adv") 
DEVICE = "cuda" if T.cuda.is_available() else "cpu"