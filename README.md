\# VeriSight — AI Image Forensics



\*\*Deepfake Image Detection Using CNN Features\*\*



VeriSight is a research prototype that uses a Convolutional Neural Network (CNN) to classify images as REAL or FAKE. It provides a simple web interface where users can upload an image and view the model's prediction and confidence scores.



\## Features



\- Upload images for deepfake classification.

\- Predict whether an image is REAL or FAKE.

\- Display probability scores for both classes.

\- Simple, interactive web interface built with Streamlit.

\- CNN-based image classification using EfficientNet-B0.

\- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.



\## Technology Stack



\- \*\*Language:\*\* Python

\- \*\*Deep Learning:\*\* PyTorch

\- \*\*Model Architecture:\*\* EfficientNet-B0

\- \*\*Image Processing:\*\* Pillow, torchvision

\- \*\*Web Interface:\*\* Streamlit

\- \*\*Data Analysis:\*\* pandas, scikit-learn



\## How It Works



1\. The user uploads an image through the VeriSight interface.

2\. The image is resized to 224 × 224 pixels and normalized.

3\. The trained EfficientNet-B0 model extracts image features and produces classification scores.

4\. The application displays the predicted class and the model's REAL and FAKE probability scores.



\## Dataset



The project uses the Deepfake image dataset from Hugging Face:



\- \*\*Dataset:\*\* \[Sowaiba01/Deepfake](https://huggingface.co/datasets/Sowaiba01/Deepfake)

\- \*\*Original images:\*\* 10,852

\- \*\*REAL images:\*\* 5,426

\- \*\*FAKE images:\*\* 5,426

\- \*\*Image format:\*\* JPEG

\- \*\*Original image dimensions:\*\* 256 × 256 pixels



The dataset was divided into training, validation, and test sets, with exact duplicate image contents checked across the splits.



\## Model Evaluation



Two EfficientNet-B0 approaches were evaluated on the same held-out test set.



| Metric | Original Model | GitHub-Style Experiment |

|---|---:|---:|

| Test images | 1,622 | 1,622 |

| Accuracy | 95.75% | 95.75% |

| REAL precision | 94.89% | 95.11% |

| REAL recall | 96.65% | 96.41% |

| Macro F1-score | 95.73% | 95.75% |



The original model is retained in the application because the experimental model did not demonstrate a meaningful overall improvement.



\### Original Model Confusion Matrix



Rows represent actual classes; columns represent predicted classes.



| Actual / Predicted | REAL | FAKE |

|---|---:|---:|

| REAL | 780 | 27 |

| FAKE | 42 | 773 |



These results are measurements on the project's test split. They do not establish performance on every image source or real-world deepfake.



\## Installation



\### 1. Clone the repository



```bash

git clone <YOUR-GITHUB-REPOSITORY-URL>

cd DeepFakeDetector

```



Replace the placeholder with your actual repository URL.



\### 2. Create and activate a virtual environment



\*\*Windows PowerShell:\*\*



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



Install the packages required by the project using your project's dependency file, if available. Otherwise, install the core packages:



```powershell

python -m pip install torch torchvision streamlit pillow pandas scikit-learn

```



\### 4. Run the application



```powershell

python -m streamlit run app.py

```



Open the local URL displayed in the terminal, usually `http://localhost:8501`.



\*\*Important:\*\* The trained model checkpoint must be available at the path expected by the application. Large dataset files and model checkpoints may need to be downloaded or provided separately.



\## Project Structure



```text

DeepFakeDetector/

├── app.py

├── model.py

├── train.py

├── evaluate\_dedup.py



