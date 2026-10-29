"""
Streamlit Page 3: Academic Dossier, Project Overview & Viva Voce Guide.
"""

import streamlit as st
from utils.ui_theme import apply_custom_theme

st.set_page_config(
    page_title="About Project & Viva Guide | AI Image Sentinel",
    page_icon="📖",
    layout="wide"
)

# Apply Global Theme
apply_custom_theme()

st.markdown("""
<div style="margin-top: 5px; margin-bottom: 20px;">
    <span class="hero-badge">🎓 Academic Deliverables</span>
    <h1 style="margin: 0; font-size: 2.2rem;">Academic Documentation & Viva Voce Dossier</h1>
    <p style="color: #94a3b8; font-size: 1rem; margin-top: 6px;">
        Formal problem statement, system architecture diagrams, mathematical formulations, and viva defense guide.
    </p>
</div>
""", unsafe_allow_html=True)

about_tab, meth_tab, viva_tab = st.tabs(["🏛️ Project Overview", "🔬 Methodology & Architecture", "🎓 Viva Voce (20 Q&A)"])

with about_tab:
    st.markdown("""
    <div class="research-card">
        <h3>1. Formal Problem Statement</h3>
        <p style="color: #cbd5e1; line-height: 1.7;">
            The rapid democratization of generative artificial intelligence technologies—specifically 
            <strong>Latent Diffusion Models</strong> (Stable Diffusion, Midjourney, DALL-E 3) and 
            <strong>Generative Adversarial Networks</strong> (StyleGAN3, ProGAN)—has eliminated 
            the perceptual boundary between optical photography and algorithmically generated imagery.
        </p>
        <p style="color: #cbd5e1; line-height: 1.7;">
            This technological shift introduces critical vulnerabilities across digital forensics, journalistic integrity, 
            electoral cybersecurity, and social trust. Conventional digital forensic watermarking and manual inspection 
            fail to generalize across modern generative paradigms. This project investigates and delivers a deep-learning-based, 
            explainable binary classification system to accurately detect synthetic visual media while exposing spatial 
            attribution heatmaps via <strong>Grad-CAM</strong>.
        </p>
    </div>
    
    <div class="research-card">
        <h3>2. Project Objectives</h3>
        <ol style="color: #cbd5e1; line-height: 1.8;">
            <li><strong>Transfer Learning:</strong> Fine-tune an ImageNet-pretrained EfficientNet-B0 architecture for high-sensitivity detection of generative visual artifacts.</li>
            <li><strong>Baseline Benchmarking:</strong> Implement and train a 4-block Custom Convolutional Neural Network from scratch to measure the transfer learning performance premium.</li>
            <li><strong>Explainable AI (XAI):</strong> Implement mathematically rigorous Grad-CAM to localize discriminative synthetic artifacts and prevent black-box opacity.</li>
            <li><strong>Zero Data Leakage:</strong> Enforce cryptographic SHA-256 deduplication to guarantee zero contamination across train, validation, and test splits.</li>
            <li><strong>Robustness Verification:</strong> Subject classifiers to real-world social media perturbations: JPEG compression, Gaussian blur, and spatial downsampling.</li>
            <li><strong>Production Deployment:</strong> Deliver an interactive, responsive Streamlit dashboard equipped for academic demonstration and viva defense.</li>
        </ol>
    </div>

    <div class="research-card">
        <h3>3. Dataset References & Citations</h3>
        <ul style="color: #cbd5e1; line-height: 1.8;">
            <li><strong>GenImage Dataset:</strong> Zhu et al., <em>"GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images"</em>, NeurIPS 2023. Covers 8 major generative models including Stable Diffusion v1.4, v1.5, Midjourney, ADM, and Wukong.</li>
            <li><strong>CIFAKE Dataset:</strong> Bird & Lotfi, <em>"CIFAKE: A Dataset for Detecting AI-Generated Synthetic Images"</em>, Data in Brief 2023. Real CIFAR-10 images paired with Stable Diffusion v1.4 generated counterparts.</li>
            <li><strong>ImageNet-1k:</strong> Deng et al., CVPR 2009. Pretraining dataset for foundational visual feature extractors.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with meth_tab:
    st.markdown("""
    <div class="research-card">
        <h3>1. Architectural Comparison</h3>
        <table style="width:100%; border-collapse: collapse; color: #cbd5e1; text-align: left;">
            <tr style="border-bottom: 2px solid #3b82f6;">
                <th style="padding: 10px;">Attribute</th>
                <th style="padding: 10px;">Baseline Custom CNN</th>
                <th style="padding: 10px;">EfficientNet-B0 (Transfer Learning)</th>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 10px; font-weight: bold;">Backbone</td>
                <td style="padding: 10px;">4-Layer Conv2D + BatchNorm + ReLU</td>
                <td style="padding: 10px;">Mobile Inverted Bottleneck Conv (MBConv)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 10px; font-weight: bold;">Pretraining</td>
                <td style="padding: 10px;">None (Initialized randomly)</td>
                <td style="padding: 10px;">ImageNet-1k Pretrained Weights</td>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 10px; font-weight: bold;">Scaling Method</td>
                <td style="padding: 10px;">Fixed Manual Dimensions</td>
                <td style="padding: 10px;">Compound Scaling (Depth, Width, Resolution)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 10px; font-weight: bold;">Grad-CAM Layer</td>
                <td style="padding: 10px;">Final Conv2D Layer (Block 4)</td>
                <td style="padding: 10px;">Final Conv Block (features[-1], 1280 channels)</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Training Strategy</td>
                <td style="padding: 10px;">End-to-end AdamW with Cosine Annealing</td>
                <td style="padding: 10px;">Two-Stage: Frozen Warmup -> Top-Block Fine-Tuning</td>
            </tr>
        </table>
    </div>

    <div class="research-card">
        <h3>2. Mathematical Formulation of Grad-CAM</h3>
        <p style="color: #cbd5e1; line-height: 1.7;">
            Grad-CAM calculates the gradient of the unnormalized class score \(y^c\) with respect to 
            feature activation map \(A^k\) of the final convolutional layer:
        </p>
        <div style="background: rgba(15, 23, 42, 0.9); padding: 15px; border-radius: 8px; text-align: center; font-size: 1.1rem; color: #60a5fa; margin: 15px 0;">
            \(\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{i,j}^k}\)
        </div>
        <p style="color: #cbd5e1; line-height: 1.7;">
            The neuron importance weights \(\alpha_k^c\) are then used to perform a weighted linear combination 
            followed by a Rectified Linear Unit (ReLU) to isolate positively influential features:
        </p>
        <div style="background: rgba(15, 23, 42, 0.9); padding: 15px; border-radius: 8px; text-align: center; font-size: 1.1rem; color: #a78bfa; margin: 15px 0;">
            \(L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)\)
        </div>
    </div>
    """, unsafe_allow_html=True)

with viva_tab:
    st.markdown("### 🎓 20 High-Yield Viva Voce Questions & Technical Answers")
    st.caption("Prepared for External Examiners, Project Evaluators, and Technical Committees.")

    viva_qa = [
        ("1. What is the fundamental difference between GANs and Diffusion Models?",
         "GANs (Generative Adversarial Networks) operate via a two-player min-max game between a Generator and Discriminator in latent space, often suffering from mode collapse. Diffusion models generate images through an iterative denoising Markov process, adding Gaussian noise forward and learning a reverse U-Net denoising schedule (DDPM/DDIM). Diffusion models achieve greater mode coverage and visual fidelity."),

        ("2. Why choose EfficientNet-B0 over ResNet-50 or VGG-16?",
         "EfficientNet utilizes compound coefficient scaling to uniformly balance network depth, width, and image resolution, outperforming traditional architectures with 4x-8x fewer parameters. EfficientNet-B0 has ~5.3M parameters compared to ResNet-50's ~25.6M and VGG-16's ~138M, achieving superior accuracy with vastly lower memory footprints and latency."),

        ("3. What is 'Data Leakage' and how did you prevent it?",
         "Data leakage occurs when information from the test or validation set contaminates training, producing falsely inflated test metrics. We enforced cryptographic SHA-256 content hashing across all images prior to splitting. Duplicate and near-duplicate images were eliminated so identical images never appear across training and test subsets."),

        ("4. Why is Grad-CAM preferable to standard Saliency Maps?",
         "Standard pixel saliency maps compute simple input-space gradients, which are notoriously noisy, sensitive to imperceptible adversarial perturbations, and lack class discrimination. Grad-CAM leverages coarse semantic representations from deep convolutional layers, combining activation maps with global-average-pooled gradient weights to yield robust, class-discriminative spatial heatmaps."),

        ("5. Why does Grad-CAM apply a ReLU activation at the final step?",
         "The ReLU activation ensures that the heatmap only visualizes features that contribute positively to the target class score. Features that belong to or support other classes (negative gradients) are suppressed, focusing the visualization strictly on discriminative evidence for the selected class."),

        ("6. What specific visual artifacts do generative models leave behind?",
         "Generative models leave distinct artifacts: (1) Spectral/frequency domain grid anomalies from transposed convolutions or upsampling latent decoders; (2) Semantic contradictions like mismatched eye reflections or anatomically incoherent fingers; (3) Hyper-smooth skin textures lacking authentic camera sensor Bayer-pattern shot noise."),

        ("7. Why did you use AdamW instead of standard Adam or SGD?",
         "Standard Adam couples L2 regularization with gradient updates, which fails to act as true weight decay in adaptive gradient algorithms. AdamW decouples weight decay directly from gradient updates, providing superior generalization, regularization, and convergence stability for deep transfer learning."),

        ("8. What is the purpose of the Cosine Annealing learning rate schedule?",
         "Cosine Annealing smoothly reduces the learning rate following a cosine curve towards a small minimum, allowing aggressive exploration in early epochs and gentle convergence to flat minima in later epochs, preventing the optimizer from bouncing out of optimal local basins."),

        ("9. Why did you employ a two-stage training strategy for EfficientNet?",
         "Pretrained weights already encode rich general visual primitives. In Stage 1, freezing the backbone prevents destructive gradient backpropagation from randomly initialized classification heads. In Stage 2, fine-tuning top blocks with a reduced learning rate allows the model to adapt high-level feature extractors specifically to synthetic visual anomalies without catastrophic forgetting."),

        ("10. What is the difference between Precision and Recall in this context?",
         "Precision measures what fraction of images flagged as 'AI-Generated' were truly AI-generated (avoiding false accusations). Recall measures what fraction of all actual AI-generated images were successfully caught (avoiding undetected deepfakes). In forensic applications, balancing both via F1-score and ROC-AUC is paramount."),

        ("11. Why is ROC-AUC a crucial metric alongside raw Accuracy?",
         "Raw accuracy can be misleading under class imbalance. ROC-AUC evaluates the classifier's discriminative ability across all possible classification decision thresholds (plotting True Positive Rate against False Positive Rate), independent of class balance or operating threshold."),

        ("12. How does JPEG compression affect AI image detection?",
         "JPEG compression applies Discrete Cosine Transform (DCT) block quantization, discarding high-frequency coefficients to reduce file size. Because synthetic image detectors often exploit subtle high-frequency latent decoding artifacts, aggressive JPEG compression (e.g. Q=50) degrades detection accuracy, highlighting the need for robustness training."),

        ("13. What is the role of ImageNet normalization constants ([0.485, 0.456, 0.406])?",
         "These values represent the channel-wise mean and standard deviation of millions of natural photographs in ImageNet. Since EfficientNet-B0 was pretrained on these statistics, normalizing input tensors to the same distribution ensures incoming activations align with the learned weights."),

        ("14. How does your model handle different input aspect ratios and resolutions?",
         "Images are safely converted to RGB, verified for data integrity, and rescaled to standard 224x224 resolution using bilinear interpolation. The Grad-CAM heatmap is computed at the feature map resolution (7x7) and subsequently upsampled using bicubic interpolation back to the original image dimensions."),

        ("15. Why does your system output a confidence score rather than a binary certainty?",
         "Deep neural network outputs represent softmax probabilities over training distributions, not mathematical certainty. Communicating scores as calibrated probabilities prevents unwarranted overconfidence and aligns with forensic best practices where human experts interpret model evidence alongside external provenance."),

        ("16. Can this detector generalize to entirely unseen future AI generators?",
         "While deep neural networks achieve high within-domain accuracy, generalization to unseen generator architectures remains an open research challenge. Generative architectures with novel latent spaces may produce unseen artifact distributions, which is why robustness evaluation and ongoing fine-tuning are essential."),

        ("17. What is Early Stopping and what hyperparameter controls it?",
         "Early stopping monitors validation loss after every epoch. If validation loss fails to improve for a predefined number of epochs ('patience', set to 4 in our system), training terminates automatically to prevent overfitting to the training set."),

        ("18. What security checks are implemented in the file upload pipeline?",
         "We enforce: (1) Extension validation (JPG, JPEG, PNG, WEBP); (2) Maximum file size limits (15 MB); (3) PIL verification and byte decoding to prevent decompression bombs; (4) Non-execution of uploaded payloads; (5) In-memory processing without writing sensitive user files to disk."),

        ("19. How does the Baseline CNN compare to EfficientNet-B0?",
         "The Baseline CNN trained from scratch demonstrates fundamental convolutional feature learning but lacks pre-trained visual representations, typically reaching lower accuracy and requiring more data. EfficientNet-B0 leverages ImageNet inductive bias, converging faster with higher precision, recall, and robustness under image perturbations."),

        ("20. What are the key future enhancements for this project?",
         "Future scope includes: (1) Dual-stream spatial and frequency (FFT / DCT) branch fusion; (2) Integration of Vision Transformers (ViT) to capture long-range token relationships; (3) Audio-visual multi-modal deepfake synchronization; (4) C2PA cryptographic content credential verification.")
    ]

    for q, a in viva_qa:
        with st.expander(f"❓ {q}"):
            st.markdown(f"""
            <div class="viva-box">
                <p style="color: #e2e8f0; line-height: 1.6; margin: 0;">{a}</p>
            </div>
            """, unsafe_allow_html=True)

