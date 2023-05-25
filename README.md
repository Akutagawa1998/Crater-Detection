# Automated Crater Detection and Classification with Machine Learning


## Introduction
This Project proposed a Metamorphic Crater Generator (MCG), a generation algorithm for crater images, and
proposes a CDA test method and a CDA training iteration strategy for YOLO V5 and other machine learning models, which realizes the fully automated
testing and iterative training progress.

## File Description

- <font color = 'blue'>MCG_REPLACE.ipynb</font>: Instruction for using MCG to do data augmentation.

- <font color = 'blue'>MCG_REPLACE.ipynb</font>: Instruction for using MCG_REPLACE to compare two group of lables for the same images (the result of detection and training sources) and find all the "craters' labels" that in the result of detection but not in the training sources. Then the MCG_REPLACE replace the cropped craters onto these position with the similar size.


- <font color = 'blue'>train_and_detect.ipynb</font>: Instruction for using YOLO V5 to do the CDA training and detecting

- <font color = 'blue'>sz5421-final-report.pdf</font>: Final report of this project

- <font color = 'blue'>counting_TP_FN.ipynb</font>:Methods for counting the number and sizes of TP, FN cases.



## Dependencies
```
# MCG requirements:

opencv-contrib-python==4.6.0.66
opencv-python==4.6.0.66
opencv-python-headless==4.6.0.66
openpyxl==3.0.10

# End ----------------------------------------

# YOLOv5 requirements"
# Usage: pip install -r requirements.txt

# Base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.1
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1  # Google Colab version
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0
protobuf<=3.20.1  # https://github.com/ultralytics/yolov5/issues/8012

# Logging -------------------------------------
tensorboard>=2.4.1
# wandb

# Plotting ------------------------------------
pandas>=1.1.4
seaborn>=0.11.0

# Export --------------------------------------
# coremltools>=4.1  # CoreML export
# onnx>=1.9.0  # ONNX export
# onnx-simplifier>=0.3.6  # ONNX simplifier
# scikit-learn==0.19.2  # CoreML quantization
# tensorflow>=2.4.1  # TFLite export
# tensorflowjs>=3.9.0  # TF.js export
# openvino-dev  # OpenVINO export

# Extras --------------------------------------
ipython  # interactive notebook
psutil  # system utilization
thop  # FLOPs computation
# albumentations>=1.0.3
# pycocotools>=2.0  # COCO mAP
# roboflow

GPU with CUDA
```

## External Code Contribution
- yolov5: This work is based on YOLO V5, you can find the website [here](https://github.com/ultralytics/yolov5) or using train_and_detect.ipynb to install this repository.

## Data Access
- Users can upload your own data through roboflow by following the instruction of train_and_detect.ipynb

- Users can also use the existing data by [google drive](https://drive.google.com/drive/folders/1qSEGnHXzX8UeNEHlAlUqsqr18hsBhHxO?usp=sharing)<br>
This link contained augemented dataset by MCG in CDA-V3-supertest, and the original dataset in CDA-V3-1

- The raw data without split can also be found in [this link](https://drive.google.com/file/d/1GpgEewb-6TB5WsKiS_GZU2-0VMFv9dMZ/view?usp=sharing)

## Getting Started
- Using MCG.ipynb to do the agumentation for your traget dataset.
- Using MCG_REPLACE to finish the Training-iteration strategy (with train_and_test.ipynb).
- Using train_and_test.ipynb to do the training and testing.
- The detailed workflow and ideas can be found in the sz4521-final-report 

## Quick look of results:
### Performance of recall rate in craters smaller than 1.5 km:
![sml_1.5.png](https://github.com/Akutagawa1998/Crater-Detection/Blob/main/sml_1.5.png)
### Compare with the SOTA:
![Table_2_new.png](https://github.com/Akutagawa1998/Crater-Detection/Blob/main/Table_2_new.png)



## Contact
ZHAO, Sihang\
sz4521@ic.ac.uk 

## Acknowledgments
Great thank to Prof. Collins and Dr. Beg <br>
Many thank to my friend Caifeng, Bo.

## License
ese-msc-2021/irp-sz4521 is licensed under the MIT License
