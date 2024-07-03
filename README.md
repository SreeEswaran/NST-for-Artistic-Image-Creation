# NST for Artistic Image Creation

This project focuses on Neural Style Transfer (NST), a technique that applies the style of one image to the content of another image, creating a new, stylized image. NST leverages deep learning models, particularly Convolutional Neural Networks (CNNs), to extract and combine the content and style features of images. This project will guide you through setting up, training, and using a neural style transfer model to create beautiful artistic images.

## Features

- **Content and Style Image Processing**: Load and preprocess images.
- **Style Transfer Model**: A CNN model for NST.
- **Training Script**: Train the model with content and style images.
- **Image Generation Script**: Generate new stylized images.
- **Results Visualization**: Visualize and save stylized images.

## How to use?

1. Clone the repository
   ```bash
   git clone https://github.com/SreeEswaran/NST-for-Artistic-Image-Creation.git
   cd NST-for-Artistic-Image-Creation
   ```

2. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```
## Train the model
   ```bash
   python script/train.py
   ```
## Generate the results
   ```bash
   python scripts/generate.py
   ```
