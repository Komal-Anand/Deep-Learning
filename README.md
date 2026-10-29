# Deep Learning Projects Repository

Welcome to the **Deep Learning** research and engineering repository by **[Komal Anand](https://github.com/Komal-Anand)**.

This repository hosts deep learning computer vision and neural architecture projects, benchmarking baseline convolutional models against modern transfer learning architectures with explainable AI (XAI).

---

## 📂 Projects Directory

### 🛡️ [AI-Generated Image Detection Using Deep Learning and Explainable AI](./AI-Generated%20Image%20Detection)

> **Final-Year B.Tech CSE (Data Science & Machine Learning) Capstone Project**  
> *Distinguishing authentic camera photographs from AI-generated visual media (Latent Diffusion Models, GANs) using **EfficientNet-B0 transfer learning**, **Baseline Custom CNN**, and **Grad-CAM** visual explainability.*

#### 🚀 Key Features:
- **Two Deep Learning Architectures**:
  - **EfficientNet-B0**: Two-stage transfer learning (classifier head warmup + top-block fine-tuning with AdamW and Cosine Annealing).
  - **Custom CNN**: 4-block convolutional network built from scratch as a comparative baseline.
- **Explainable AI (Grad-CAM)**: PyTorch hook-based gradient activation maps highlighting discriminative synthetic artifacts.
- **Multi-Modal Forensic Suite**:
  - **2D Fast Fourier Transform (FFT)**: Log magnitude spectrum identifying high-frequency deconvolution checkerboard patterns.
  - **Error Level Analysis (ELA)**: 90% JPEG compression delta matrix exposing localized manipulation disparities.
  - **Metadata & AI Prompt Inspector**: Parses PNG/WebP chunk data for embedded generative prompts, seeds, and camera EXIF.
- **Interactive Web App**: Dark navy research dashboard built in Streamlit featuring a **"Human vs. AI" Visual Turing Challenge** and one-click downloadable **Forensic Audit Certificates**.
- **Rigorous Data Engineering**: Enforces cryptographic SHA-256 deduplication to ensure zero data leakage across train, val, and test splits.
- **Robustness Benchmark**: Evaluated under real-world social media perturbations: JPEG compression ($Q=50$), Gaussian blur, and spatial downscaling.

#### 📖 Quick Links:
- 📁 **[Project Source Code & Instructions](./AI-Generated%20Image%20Detection)**
- 📊 **[Evaluation Reports & Visualizations](./AI-Generated%20Image%20Detection/reports)**
- 📄 **[Comprehensive Academic Project Dossier](./AI-Generated%20Image%20Detection/docs/FINAL_YEAR_PROJECT_REPORT.md)**
- 🎓 **[20 Viva Voce Questions & Answers](./AI-Generated%20Image%20Detection/docs/FINAL_YEAR_PROJECT_REPORT.md#10-twenty-likely-viva-questions--technical-answers)**

---

## 🛠️ Technology Stack
- **Frameworks**: PyTorch, Torchvision, Streamlit
- **Computer Vision**: OpenCV, Pillow
- **Data Science**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Version Control**: Git & GitHub

---

## 👤 Author
**Komal Anand**  
- GitHub: [@Komal-Anand](https://github.com/Komal-Anand)  
- Specialization: B.Tech CSE (Data Science & Machine Learning)
