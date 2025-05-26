# demo_gradcam.py
from grad_cam import demo

if __name__ == "__main__":
    img_path   = "dataset/Training/notumor/Tr-no_0010.jpg"
    model_path = "models/brain_tumor.h5"
    last_conv  = "conv2d_1"

    demo(img_path=img_path, model_path=model_path, last_conv_layer=last_conv)
