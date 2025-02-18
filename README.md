# Black & White Image Colorization

This project is a Streamlit-based web application that colorizes black & white images using a deep learning model based on OpenCV's DNN module.

## Features
- Upload a black & white image (JPG, PNG, or JPEG format).
- Automatically colorizes the image using a pre-trained Caffe model.
- Displays both the original and colorized images side by side.
- Allows users to download the colorized image.

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install streamlit numpy opencv-python
```

## Model Files
Download the necessary model files and place them in the `model/` directory inside your project folder:
- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

Update the `DIR` variable in the script to point to the correct directory containing these files.

