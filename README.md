
## Project Description

# Software Dependencies
The following software dependencies were used to reproduce the experiments in this study:
## Python Version:
- Python 3.8
## Deep Learning Frameworks:
- TensorFlow 2.6.0
- PyTorch 1.9.0
## Libraries:
- Numpy 1.21.2
- Matplotlib 3.4.2
- OpenCV 4.5.1
## Hardware:
- L40S GPU (CUDA 11.2)
- 48 GB VRAM

This project uses two datasets: the generated dataset (`generated`) and the source dataset (`source`) to train and evaluate the model. Below is a detailed explanation of the dataset files and how to use them.

## Dataset File Description

### Dataset Files

- **generated_2.7z.001, generated_2.7z.002, generated_2.7z.003, generated_2.7z.004**:  
  These files contain the processed and generated dataset. They include data used for training the generated model. You can extract them using 7-Zip or any other software that supports the `.7z` format. The extracted data will be used to train the generated model.

- **source_2.7z.001, source_2.7z.002, source_2.7z.003**:  
  These files contain the original dataset, typically used as input data for model training. You can use these files to compare the generated dataset (`generated`) to the original dataset. After extraction, this data will be used to train the source model or for comparison experiments.

### Extracting the Dataset

Since the dataset is split into multiple `.7z` parts, you need software that supports split archive formats to extract these files. We recommend using [7-Zip](https://www.7-zip.org/) for extraction:

1. Place all `.7z.001`, `.7z.002`, `.7z.003`, and `.7z.004` files in the same folder.
2. Use 7-Zip to extract `generated_2.7z.001` or `source_2.7z.001`. The software will automatically combine all parts and extract the complete folder.


