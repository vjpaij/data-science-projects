## 1\. Project Overview

Build an Image Generation System that can generate realistic images using Generative Adversarial Networks (GANs) or Diffusion Models (if GANs underperform). 

* * *

## 2\. Tools & Technologies

| Category | Tools/Libraries |
| --- | --- |
| Framework | PyTorch (with CUDA) |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit (preferred) / Flask |
| Version Control | Git, GitHub |
| Cloud (Optional) | Google Colab (Free GPU) / AWS SageMaker |

* * *

## 3\. Dataset Dictionary

We'll use the **CIFAR-10** dataset (small but good for prototyping GANs).  
Alternatively, **CelebA** (for faces) or **MNIST** (for digits) can be used.

| Column | Description |
| --- | --- |
| Images | 32x32 RGB images (CIFAR-10) |
| Labels | 10 classes (airplane, car, bird, etc.) |

* * *

## 4\. How to Run

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Jupyter notebook: run image-generate.ipynb
4. Open and run the Main Analysis script: 'python image_train.py'
5. For Streamlit: streamlit run 'streamlit run image_app.py'

* * *

### 5\. Additional Resources (Udemy Video Courses)    
*   GANs & Diffusion Models	- https://www.udemy.com/course/generative-adversarial-networks/

* * *