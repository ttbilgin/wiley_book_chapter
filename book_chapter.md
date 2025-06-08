# Part II: Core Generative Techniques

## 1. Generative Adversarial Networks (GANs)
### 1.1 Fundamentals of GANs
### 1.2 Variants of GANs for medical imaging
### 1.3 GANs in brain imaging and neurological applications

## 2. Variational Autoencoders (VAEs)
### 2.1 Basics of VAEs
### 2.2 Applications of VAEs in neuroscience
### 2.3 Comparing VAEs and GANs for healthcare research

## 3. Emerging Generative Techniques
### 3.1 Diffusion models in medical applications
### 3.2 Neural ODEs for brain disease modeling
### 3.3 Reinforcement learning and generative applications

---

# ABSTRACT

Neurological diseases pose major challenges in modern healthcare, requiring advanced techniques for diagnosis, monitoring, and treatment. In response, artificial intelligence is emerging as a valuable tool, with generative modeling techniques offering promising advancements in medical imaging, disease monitoring, and treatment optimization.

This chapter provides a comprehensive overview of five core generative techniques transforming neurological disease research: Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), diffusion models, neural ordinary differential equations (Neural ODEs), and reinforcement learning (RL).

**GANs** excel at producing high-fidelity brain images through adversarial training between generator and discriminator networks. Variants like CycleGAN, Pix2Pix, and WGAN-GP have demonstrated success in data augmentation, cross-modal synthesis (MRI↔PET), and brain tumor segmentation, though training stability remains challenging.

**VAEs** provide a probabilistic framework for learning latent representations of neurological data. Their explicit probabilistic structure makes them valuable for anomaly detection, disease progression modeling, and interpretable feature extraction, while maintaining superior training stability compared to GANs.

**Diffusion models** represent the newest advancement in medical image generation, offering superior quality-diversity trade-offs. These models excel at generating high-resolution 3D brain MRI images and completing missing longitudinal data, directly addressing data scarcity in neurological research.

**Neural ODEs** provide a principled approach to modeling continuous-time evolution of neurological systems. They are uniquely suited for capturing brain dynamics from fMRI data and modeling disease progression with irregular sampling, though practical applications remain limited.

**Reinforcement learning** enables adaptive, closed-loop decision-making for personalized neurological care. RL has shown success in optimizing deep brain stimulation for Parkinson's disease, adaptive rehabilitation protocols, and personalized cognitive training systems.

Despite these advances, clinical integration faces significant challenges including data quality and scarcity, model interpretability concerns, regulatory compliance requirements, and the need for medical-specific evaluation metrics. Ethical considerations such as bias mitigation and patient privacy protection are crucial for responsible deployment.

Comparative analysis reveals complementary strengths: GANs produce sharp images but may have training instability; VAEs offer stable training and interpretable latents but blurrier outputs; diffusion models achieve superior quality but require significant computational resources; Neural ODEs provide principled temporal modeling but limited practical applications; and RL enables adaptive interventions but faces interpretability challenges.

The chapter examines hybrid approaches combining multiple techniques, medical-specific evaluation frameworks, and standardized validation protocols. Through comprehensive analysis of these five generative techniques, this chapter demonstrates how they form a powerful toolkit for addressing neurological disease challenges and advancing personalized, data-driven healthcare solutions.

---

# 1. Generative Adversarial Networks (GANs) in Medical and Brain Imaging

## 1.1 Fundamentals of GANs

Generative adversarial networks (GANs) consist of two neural networks – a *generator* and a *discriminator* – trained in an adversarial (minimax) game. The generator $G$ tries to produce realistic images from random noise $z\sim p_z$, while the discriminator $D$ aims to distinguish real images $x\sim p_{\text{data}}$ from $G$'s outputs. Typically the objective is given by the minimax loss:

$$
\min_G \max_D V(G,D) \;=\; E_{x\sim p_{\text{data}}(x)}[\log D(x)] \;+\; E_{z\sim p_z(z)}[\log(1-D(G(z)))].
$$

Under this loss, $D$ is trained to maximize the probability of correctly classifying real and fake samples, while $G$ is trained to minimize $\log(1-D(G(z)))$. In practice, non-saturating losses or variants (e.g. least-squares loss) are often used to stabilize training. Conditional GANs (cGANs) introduce labels or images as conditioning inputs, allowing generation of outputs that satisfy constraints (e.g. *Pix2Pix* for paired image translation). 

Unpaired translation GANs (e.g. *CycleGAN*) use cycle-consistency losses to enable mapping between two image domains without paired examples. The complete CycleGAN cycle consistency loss includes both forward and backward cycles:

$$
L_{\text{cyc}}(G,F) = E_{x\sim p_{\text{data}}(x)}[\|F(G(x)) - x\|_1] + E_{y\sim p_{\text{data}}(y)}[\|G(F(y)) - y\|_1]
$$


where $G: X \rightarrow Y$ and $F: Y \rightarrow X$ are the mappings between domains $X$ and $Y$. This bidirectional constraint ensures both $x \rightarrow y \rightarrow x \approx x$ and $y \rightarrow x \rightarrow y \approx y$, which is essential for preserving anatomical structures in medical imaging applications.

Standard GAN training can be unstable (mode collapse, vanishing gradients), so improved variants have been proposed. Wasserstein GANs (WGANs) replace the Jensen–Shannon divergence with the Earth-Mover (Wasserstein) distance, which yields smoother gradients and mitigates mode collapse. WGAN enforces a Lipschitz constraint (e.g. weight clipping) to compute the Wasserstein distance between real and generated distributions. The WGAN-GP variant adds a gradient penalty to enforce the Lipschitz condition without weight clipping. These loss modifications improve training stability at the cost of additional complexity. 

Overall, GANs are versatile generative models: they can produce synthetic images that mimic a target distribution, and under adversarial training the generator learns to capture the real data distribution. Mathematically, GANs attempt to solve the two-player game

$$\min_G\max_D E_{x\sim p_{\text{data}}}[\log D(x)] + E_{z\sim p_z}[\log(1-D(G(z)))]$$

often augmented with additional terms (e.g. conditional losses, cycle consistency, reconstruction loss). These core formulations enable applications across medical imaging tasks – from generating realistic MR or CT images to augment datasets and to performing image-to-image translations.

## 1.2 Variants of GANs for Medical Imaging

In recent years, many GAN architectures have been adapted to medical imaging tasks. Typical variants include **DCGAN** (deep convolutional GAN for realistic image synthesis), **Pix2Pix** (paired image-to-image translation using cGAN), **CycleGAN** (unpaired translation with cycle consistency), **WGAN** (Wasserstein GAN with improved stability), **WGAN-GP** (with gradient penalty), **StyleGAN** (progressive growing and style-based generation for high-resolution images), and **Pix2PixHD**, **ProGAN**, etc. These have been applied to all major modalities (MRI, CT, X-ray, PET, ultrasound) and tasks like image synthesis, reconstruction, segmentation, and enhancement.

For example, *CycleGAN* has been extensively used to convert images across modalities when paired data are scarce. Zhu et al.'s CycleGAN learns inverse mappings $G_{X\to Y}$ and $G_{Y\to X}$ with a cycle-consistency loss. In medical imaging, CycleGAN has been used to synthesize CT from MRI for radiotherapy planning, or to translate between different MRI contrasts. *Pix2Pix* (a cGAN) has been applied to segmentation by treating the segmentation mask as the target image. For instance, Cai et al. (2024) applied a Pix2Pix-based GAN to lung CT segmentation: the model translated raw CT slices to binary lung masks and achieved higher accuracy than a standard U-Net baseline.

![image](https://github.com/user-attachments/assets/7e9caf58-e559-4e94-bee6-ed741a2109a3)

**Fig. 1. Overview of the BPGAN (Zhang et al. (2022))**

Similarly, *WGAN* and *WGAN-GP* have been used for reconstructing undersampled MRI or generating high-quality synthetic images without mode collapse. For example, Zhang et al. (2022) proposed a 3D WGAN-GP variant (BPGAN) to generate PET images from MRI scans in Alzheimer's research. Their generator was a 3D U-Net, and WGAN-GP stabilization allowed the synthetic PET to improve multi-class AD diagnosis by approximately 1% over MRI alone. In image enhancement, GANs have enabled super-resolution: Costa de Farias et al. (2021) introduced a lesion-focused GAN (GAN-CIRCLE with Spatial Pyramid Pooling) to super-resolve lung tumor patches in CT images. At 2× resolution, their GAN produced sharper, more detailed images than other SR methods. They showed that radiomic features extracted from GAN-super-resolved images were more robust to quantization, indicating improved image quality.

More recent high-resolution generators like *StyleGAN* and progressive GANs have also been used in medical imaging. For brain tumor imaging, Akbar et al. (2024) evaluated StyleGAN and ProGAN for synthesizing tumor MRI scans. They found that segmentation networks trained on GAN-generated brain tumor images achieved 80–90% of the Dice score of models trained on real images. While this demonstrates the potential of synthetic data augmentation, it also highlights the performance gap that currently exists between synthetic and real data training approaches. Similarly, StyleGAN has been applied to generate photorealistic histopathology images for data augmentation and style transfer.

Table 1 summarizes representative GAN-based medical imaging studies, indicating the GAN variant, imaging modality, task, and reported performance:

### **Table 1. Summary of Representative GAN-based Medical Imaging Studies**

| Study (Year)                    | GAN Variant                               | Modality                 | Task                                              | Key Results / Performance                                                                                                                         |
| ------------------------------- | ----------------------------------------- | ------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gulakala *et al.* (2022)        | U-Net GAN (ResUNet with skip connections) | Chest X-ray              | Data augmentation + COVID-19 classification       | Achieved 99.2% test accuracy for 3-class detection, though 98.7% of COVID training data was synthetically generated from only 27 real images                                      |
| Zhang *et al.* (2022)           | 3D cGAN (WGAN-GP)                         | Brain MRI → PET          | Synthesize FDG-PET from T1 MRI (ADNI dataset)     | Synthetic PET had high similarity (↑PSNR, SSIM) to real; multi-class AD diagnosis accuracy improved by ~1% using synthetic PET                   |
| Costa de Farias *et al.* (2021) | Lesion-focused GAN-CIRCLE (GAN+SPP)       | CT (lung tumors)         | Lesion-centric super-resolution (2×, 4× SR)       | At 2× SR, GAN produced higher perceptual quality (less blurring) than other SR methods; improved robustness of lung tumor radiomic features       |
| Akbar *et al.* (2024)           | Progressive GAN / StyleGAN 1-3            | Brain MRI (glioma scans) | Synthetic image generation for tumor segmentation | Segmentation networks trained on synthetic images reached 80–90% of the Dice score of real-data training, indicating remaining performance gaps |
| Cai *et al.* (2024)             | Pix2Pix (cGAN)                            | CT (thoracic)            | Lung segmentation via image translation           | Pix2Pix translated input CT to lung masks; achieved better accuracy than a U-Net baseline (Dice and F-measure improved)                           |

Each of the above studies demonstrates how specific GAN variants serve different goals: DCGAN/U-Net GANs for generative augmentation, cGANs (Pix2Pix) for segmentation, CycleGANs for modality translation, and WGAN-based models for stabilizing training in reconstruction tasks. These works report performance improvements using adversarial losses, though it's important to note that most applications remain in research phases rather than clinical deployment. The performance gains should be interpreted within the context of specific datasets and experimental conditions rather than as indicators of clinical readiness.

## 1.3 GANs in Brain Imaging and Neurological Applications

GANs have been extensively applied to neuroimaging, addressing tasks in Alzheimer's disease (AD), Parkinson's disease (PD), epilepsy, stroke, and brain tumors, using structural (MRI, DTI), functional (fMRI, PET), and even EEG data. A major use-case is **disease classification and harmonization**. For example, Zhang *et al.* (2022) used a WGAN-GP to augment MRI data for multiclass AD diagnosis. By generating additional samples for underrepresented classes, their model improved classification performance across ADNI subtypes. Similarly, GAN-based *harmonization* networks have been employed to reduce scanner/site differences. Harmonizing multi-site MRI (ADNI/AIBL/OASIS) with a GAN improved AD vs. control classification accuracy by aligning distributions, though the practical benefits depend on the specific imaging protocols and populations studied.

GANs also facilitate **missing-modality synthesis**. Zhang *et al.* (2022)'s BPGAN generated full-resolution PET images from available MRI scans. Their synthetic PET retained important AD-related features: combined MRI+synthetic PET improved diagnosis accuracy by approximately 1% compared to MRI alone. This demonstrates GANs' potential to "impute" missing modalities, enabling multi-modal analysis even when PET data is unavailable due to cost or radiation concerns. The cycle-consistency paradigm has likewise been used to synthesize MRI from CT or vice versa in brain imaging applications.

Another emerging use is **disease progression modeling**. Bossa *et al.* (2024) trained a 3D GAN on longitudinal amyloid PET scans (ADNI) to learn a latent representation of amyloid burden. They then built an ordinary-differential-equation model in the latent space to predict amyloid accumulation over time. The GAN latent features accurately predicted global amyloid SUVR with low error, and synthetic PET trajectories reflected actual 4-year amyloid changes. This novel pipeline demonstrates how GANs can capture disease dynamics and simulate progression in neurological disorders, though validation in larger, more diverse populations is needed.

**Synthetic data generation and augmentation** is vital in brain imaging, where labeled datasets are limited. For example, Akbar *et al.* (2024) showed that GAN-generated brain tumor MRIs could train segmentation models to achieve 80–90% of the performance obtained with real images. While this represents substantial progress, it also highlights the performance gap that must be addressed before synthetic data can fully substitute for real clinical data. In brain tumor classification, researchers have used GANs to create synthetic glioma images or to sample from continuous latent spaces (e.g. StyleGAN) that reflect different tumor phenotypes. These synthetic images can supplement scarce clinical data and improve model generalization, though careful validation is required to ensure clinical relevance.

Finally, GANs assist in **denoising and resolution enhancement** in neuroimaging. Research groups have used GANs to enhance low-dose or low-resolution scans, potentially enabling dose reduction or faster acquisition times. GANs have also been applied to improve fMRI preprocessing (denoising, super-resolution) and to synthesize diffusion MRI contrasts, although these applications remain largely experimental.

In summary, recent studies (2019–2024) illustrate GANs' versatility in brain-related applications: from improving AD/PD diagnosis (via augmentation and harmonization) to modeling progression (amyloid PET trajectories) and completing missing data (MRI↔PET synthesis). GANs also show promise for generating realistic EEG/seizure signals for epilepsy research, though imaging domains dominate current applications. In all cases, GAN-based methods report enhanced downstream performance (classification accuracy, segmentation metrics, etc.) and open new possibilities for dealing with data scarcity and heterogeneity in neurological imaging. However, it's crucial to recognize that most of these applications are still in research phases, with significant work needed before clinical translation can occur.

Despite their promising applications, GANs in medical imaging face several important challenges that must be addressed for successful clinical translation. **Evaluation metrics** represent a significant concern, as traditional computer vision metrics like PSNR and SSIM may not adequately capture medical image quality. Medical imaging requires domain-specific evaluation criteria that consider diagnostic relevance, anatomical accuracy, and clinical utility rather than just visual similarity.

**Data distribution concerns** arise when training sets are small or unrepresentative, as GANs may memorize training examples rather than learning generalizable features. This is particularly problematic in medical imaging where acquiring large, diverse datasets is challenging due to privacy constraints, rare conditions, and expensive imaging procedures. Researchers must carefully validate that synthetic images capture true population variability rather than artifacts of limited training data.

**Regulatory and validation requirements** for medical AI present additional hurdles. Unlike other AI applications, medical imaging GANs must demonstrate safety, efficacy, and generalizability across diverse patient populations before clinical deployment. Current research achievements, while impressive, require extensive validation studies and regulatory approval processes that can take years to complete.

**Clinical integration challenges** include ensuring that synthetic data preserves diagnostically relevant features, avoiding introduction of artifacts that could mislead clinical interpretation, and maintaining consistency with established imaging protocols and analysis pipelines. The performance gaps observed between synthetic and real data training (such as the 80-90% Dice scores mentioned earlier) illustrate that significant technical advances are still needed.

Future research directions should focus on developing **medical-specific evaluation frameworks**, creating **larger and more diverse training datasets**, establishing **standardized validation protocols** for medical AI applications, and building **explainable GAN architectures** that allow clinicians to understand and trust synthetic image generation processes. Hybrid approaches combining GANs with other generative models may also offer promising avenues for addressing current limitations.

---

# 2. Variational Autoencoders in Medical Imaging

## 2.1. Basics of VAEs

A Variational Autoencoder (VAE) is a generative deep model combining an **encoder-decoder** architecture with a probabilistic latent space. The encoder maps each input (e.g. an image **x**) to parameters of a latent distribution $q(z|x)$ (typically Gaussian with mean μ(x) and variance σ²(x)). A latent vector **z** is sampled from this distribution and fed to the decoder, which reconstructs the input. Training maximizes the *evidence lower bound* (ELBO), comprising a reconstruction loss plus a Kullback–Leibler (KL) divergence between $q(z|x)$ and a chosen prior $p(z)$ (usually $N(0,I)$). The KL term regularizes the latent space and prevents overfitting. In β-VAEs, a weight β scales the KL term, trading off reconstruction fidelity against latent smoothness. After training, sampling z from the prior and decoding produces new synthetic images that resemble real data. In summary, VAEs learn a compressed **latent representation** of data (often lower-dimensional), while enabling generation of new samples via the probabilistic latent space.

Key components include:

* **Encoder:** Neural network producing latent parameters ($\mu(x)$, $\sigma(x)$) from input. 
* **Latent space:** Continuous probabilistic embedding; in a standard VAE each latent dimension is Gaussian.
* **Decoder:** Neural network that maps a sampled z back to the data space to reconstruct $\hat x$.
* **Loss (ELBO):** $$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \mathrm{KL}[q(z|x)\|p(z)]$$. 

For implementation, this is often approximated as: $$\mathcal{L} \approx \frac{1}{2\sigma^2}\|x - \hat x\|^2 + \beta\,\mathrm{KL}(q(z|x)\|N(0,I)) + \text{constants}$$, where the scaling factor $\frac{1}{2\sigma^2}$ depends on the assumed decoder output variance and β controls the KL regularization strength.

VAEs are widely used in medical imaging because they can augment datasets by generating realistic synthetic examples. Their probabilistic nature and smooth latent space facilitate tasks like anomaly detection (by measuring likelihood under the model) and feature extraction.

## 2.2. Applications of VAEs in Neuroscience

VAEs have been applied across neuroscience modalities (MRI, fMRI, PET, EEG) for diverse tasks (classification, progression modeling, synthesis, anomaly detection, dimensionality reduction). As shown in Table 2 Recent studies include:

### **Table 2. Summary of Applications of VAEs in Neuroscience Studies**

| Study (Year)              | Data Type (Modality)      | Task / Disease                                  | VAE Model                              | Key Findings                                                                                                                                                                                                                                                                                      |
| ------------------------- | ------------------------- | ----------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kumar *et al.* (2022)     | Multimodal MRI (T1, T2)   | Alzheimer's disease (normative modeling)        | Multimodal VAE                         | Proposed a **multi-modal VAE** on joint T1/T2 brain volumes to learn healthy controls' distribution. Testing on AD patients, the method produced deviation maps more sensitive to disease stage and better correlated with cognition than a unimodal baseline.                               |
| He *et al.* (2024)        | Longitudinal MRI (ADNI)   | Alzheimer's progression (trajectory prediction) | Conditional VAE (CVAE)                 | Used a CVAE to generate individualized future MRI scans given baseline scan, age, and diagnosis. The model could **extrapolate brain changes** and produced synthetic data useful for downstream tasks (anomaly detection, classification).                                                     |
| Hassanaly *et al.* (2024) | 3D FDG-PET                | Dementia (unsupervised anomaly detection)       | VAE ensemble (17 variants)             | Benchmarked 15 VAE variants (plus vanilla AE/VAE) for detecting anomalies in PET related to Alzheimer's. Nine variants yielded accurate reconstructions and healthy-looking images. **Many VAE variants generalized well**, but **none outperformed the basic VAE/AE** in anomaly detection. |
| Kim *et al.* (2021)       | Resting-state fMRI        | Individual identification (no disease)          | β-VAE                                  | Applied a β-VAE on large HCP rs-fMRI data. The **latent features** learned were highly informative: they could *reliably identify subjects* from their fMRI patterns with ~99% accuracy, demonstrating that the VAE captured individual brain signatures.                                     |
| Han *et al.* (2019)       | Task fMRI (visual cortex) | Visual encoding/decoding                        | Standard VAE                           | Trained a VAE to model V1 visual cortex responses to natural images. The encoder-decoder learned to **reconstruct stimulus images from fMRI** and vice versa, serving as an unsupervised model of the visual processing pathway.                                                             |
| Yue *et al.* (2023)       | EEG (resting state)       | Obesity vs. lean subject classification         | VAE feature extractor + CNN classifier | Introduced a VAE to extract subject-invariant EEG features for obesity classification. Using these latent features in a 1D-CNN classifier yielded **significantly higher accuracy** for distinguishing obese and lean groups, outperforming conventional classifiers.                        |

These examples illustrate key use-cases: VAEs are used to **extract low-dimensional representations** of brain data (for classification or clustering), to **synthesize or predict images** (augmenting or simulating longitudinal scans), and to **detect anomalies** by modeling normal anatomy. For instance, Kumar et al. used a multimodal VAE to capture complex structural changes across MRI modalities. VAE-based generators like the CVAE in He et al. can predict future brain scans, aiding disease modeling. In fMRI, unsupervised VAEs have been shown to uncover meaningful latent factors, such as individual-specific network features and visual cortex representations. Overall, VAEs have proven valuable in neuroscience for **dimensionality reduction** and **feature learning**, especially when probabilistic embeddings and reconstruction losses align with clinical data characteristics.

## 2.3. Comparing VAEs and GANs in Healthcare Research

Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) are two dominant deep generative frameworks in medical AI. Recent literature (2019–2024) highlights complementary strengths and weaknesses for healthcare tasks, though diffusion models are increasingly challenging both approaches:

* **Image Synthesis:** GANs typically produce *higher-fidelity, sharper images* by training a generator against a discriminator, but they can suffer from training instability and mode collapse. VAEs generate more *diverse* outputs (covering more modes) and are easy to train, but their reconstructions tend to be **blurry** due to the Gaussian prior and KL regularization. As noted in recent reviews, *"VAEs outperform GANs in terms of output diversity and are free of mode collapse. However… \[they have a] tendency to often produce blurry and hazy output images"*. Hybrid models (e.g. VAE-GAN) attempt to combine the VAE's latent smoothness with GAN's image sharpness. However, **diffusion models now often outperform both** in medical imaging applications, achieving better diversity and fidelity.

* **Data Imputation:** GAN-based methods like **GAIN** (Generative Adversarial Imputation Nets) have shown strong performance for imputing missing clinical data, especially at high missingness rates. VAE-based imputers (e.g. MIWAE, HI-VAE) can also handle mixed-type missing data and model uncertainty, but benchmarks indicate their accuracy may degrade when data are highly heterogeneous or missingness is extreme. In practice, GANs offer **fast, accurate completion** for EHR/time-series, whereas VAEs provide principled uncertainty but can require careful tuning.

* **Disease Progression Modeling:** VAEs (especially *conditional VAEs*) naturally model distributions and can generate plausible future observations. For example, VAEs have been used to simulate disease progression by sampling from learned trajectories. GANs can likewise generate longitudinal images (e.g. by conditioning on time), but require more complex adversarial setups. GANs may produce more realistic individual scans, but VAEs provide explicit likelihoods and smooth interpolation over disease stages.

* **Representation Learning & Interpretability:** VAEs yield an explicit latent space with a known prior, making them more interpretable: the latent dimensions capture meaningful features of the data. In contrast, GAN latent spaces lack an explicit probability model and are harder to interrogate. As a result, VAEs are often preferred when a low-dimensional embedding or feature attribution is needed (e.g. for biomarker discovery).

### **Table 3. Comparative Analysis of VAE and GAN Applications in Healthcare**

| **Application/Task**               | **VAE (Advantages)**                                                                                     | **GAN (Advantages)**                                                                      | **Remarks**                                                                                                                                            |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Image Synthesis (Augmentation)** | - High sample diversity; no mode collapse<br/>- Stable, likelihood-based training                        | - Produces sharp, realistic images<br/>- Well-established in imaging (many architectures) | VAEs produce blurrier images; GANs may ignore some modes. Diffusion models often outperform both in recent medical imaging studies.                  |
| **Data Imputation (Missing Data)** | - Provides probabilistic estimates (uncertainty) for imputations                                         | - High accuracy in practice (e.g. GAIN)<br/>- Fast convergence                            | VAE-based imputers exist (MIWAE, HI-VAE) but can falter with complex EHR patterns. GAN imputers handle heterogeneous mixes well.                       |
| **Disease Progression Modeling**   | - Can model distributions of trajectories; handles uncertainty smoothly (e.g. CVAE)                      | - Can generate photo-realistic future states with adversarial loss                        | VAEs yield probabilistic forecasts useful for long-term predictions; GANs yield sharper, potentially more realistic snapshots but need careful design. |
| **Representation Learning**        | - Learns a smooth latent space reflecting data variation<br/>- Latents are interpretable (feature-based) | - No explicit latent inference (latent code is adversarially constrained)                 | VAEs explicitly encode data to z∼p(z\|x), facilitating analysis; GANs do not encode input to a latent by default (except autoencoder variants).        |
| **Interpretability**               | - Latent variables have statistical meaning; disentanglement possible                                    | - Limited interpretability; latent space not semantically regularized                     | The smooth VAE latent allows tracing how factors change; GAN's adversarial training makes disentanglement difficult.                                   |

In summary, **VAEs** are favored when robust, diverse sampling and interpretable embeddings are important. **GANs** excel when generating the highest-quality images or realistic samples is critical. Recent reviews emphasize this complementary nature. For example, VAEs are praised for *"ease of training and good coverage of the data distribution"*, whereas GANs are celebrated for image *fidelity*. In healthcare, the choice often depends on the application: e.g. GANs are widely used for **image synthesis/augmentation** (MRI, histology, etc.), while VAEs underpin tasks like **anomaly detection**, **latent factor analysis**, and **probabilistic progression modeling**. However, **diffusion models are increasingly preferred** for medical image synthesis due to their superior quality-diversity trade-offs.

Despite their promising applications, VAEs in medical imaging face several important challenges. **Evaluation metrics** for medical applications often require domain-specific criteria beyond traditional reconstruction losses. **Data scarcity** in medical domains means VAEs may struggle to capture true population variability, especially for rare conditions. **Regulatory validation** for clinical deployment requires extensive testing and approval processes that can take years.

Future research should focus on **multimodal VAEs** that integrate different imaging modalities, **disentangled representations** for better interpretability, and **hybrid approaches** combining VAEs with diffusion models or other generative techniques. The field is also moving toward **conditional VAEs** that can incorporate clinical metadata for more targeted synthesis and **federated learning** approaches that can train on distributed medical datasets while preserving privacy.


---

## **3. Emerging Generative Techniques**

Deep learning-based generative models—such as diffusion models, neural ordinary differential equations (Neural ODEs), and reinforcement learning (RL)—are offering new possibilities for diagnosing, monitoring, and treating neurological disorders. However, one of the most pressing limitations in this field lies in the restricted number of available medical imaging datasets. Clinical datasets often contain only a few thousand samples, limiting the generalizability of models and hindering their application in real-world clinical settings. To overcome these limitations, synthetic data generation has emerged as a highly promising avenue. By using diffusion models, realistic brain images can be generated to diversify and expand existing datasets, leading to improvements in model performance.

In this chapter, we present a brief review of the use of diffusion models, Neural ODEs, and reinforcement learning in addressing major neurological disorders, including Alzheimer's disease, Parkinson's disease, epilepsy, multiple sclerosis (MS), and amyotrophic lateral sclerosis (ALS). We examine key publications from the last five years, highlighting methodological innovations, application domains, and performance benchmarks (Table 4.).

### **Table 4. Summary of Reviewed Studies**

| Neurological Disorder | Study (Year)               | Methodology                        | Data Type                  | Key Findings                                                                |
| --------------------- | -------------------------- | ---------------------------------- | -------------------------- | --------------------------------------------------------------------------- |
| Alzheimer's Disease   | Chen et al. (2023)         | Multi-feature Fusion Network      | EEG (Resting State)        | Integrated spectral, temporal, and spatial features for improved AD diagnosis |
|                       | Yuan et al. (2024)         | 3D Diffusion Model (ReMiND)       | MRI (ADNI longitudinal)    | Imputed missing MRIs; accurately predicted hippocampal atrophy               |
|                       | Bossa & Sahli (2023)       | Multidimensional ODE Model        | Clinical Data              | Modeled AD progression using conventional differential equations              |
|                       | Pinaya et al. (2022)       | Latent Diffusion Model            | MRI (UK Biobank)           | Generated 100,000+ synthetic brain MRIs from 31,740 real images             |
|                       | Zheng et al. (2023)        | Integrated Feature Analysis       | EEG (Resting State)        | Combined spectrum, complexity, and synchronization features for AD diagnosis  |
| Parkinson's Disease   | Cho et al. (2024)          | RL (TD3, PPO, A2C, SAC)           | DBS Simulations            | Optimized closed-loop DBS with significant energy reduction                  |
|                       | Lian et al. (2024)         | Multi-modal Graph Networks        | Clinical Multimodal Data   | Personalized progression prediction using graph neural networks              |
|                       | Yin et al. (2024)          | Machine Learning Gait Analysis    | Gait Movement Data         | Early-stage PD detection using wearable sensors and ML algorithms           |
| Epilepsy              | Kuang et al. (2024)        | GCN + LSTM Networks               | Multi-channel EEG          | Combined spatial and temporal modeling for seizure prediction                |
|                       | Liu et al. (2024)          | Pseudo-3D CNN                     | EEG Signals                | Improved seizure prediction using 3D convolutional approaches               |
| Multiple Sclerosis    | Valencia et al. (2022)     | Conditional 3D GAN (Pix2Pix)     | MRI (T2-FLAIR → T1-w)      | Improved sensitivity for new T2 lesion detection using synthetic T1 images   |
|                       | Salem et al. (2020)        | U-Net GAN                         | Multi-modal MRI            | 83% true positive rate, 9.36% false positive rate for lesion detection      |
|                       | Puglisi et al. (2024)      | Latent Diffusion (BrLP model)    | T1-weighted MRI (11,730)   | 22% increase in volumetric accuracy, 43% improvement in image similarity     |
| ALS                   | Regondi et al. (2025)      | HiFi-GAN                          | Voice/Speech Signals       | High-quality synthetic voice generation for personalized voice banking       |
|                       | Sengur et al. (2017)       | DCGAN + Reinforcement Learning    | EMG Signals                | 96.80% classification accuracy for ALS vs. control using synthetic EMG data  |
|                       | Hazra & Byun (2021)        | SynSigGAN (BiLSTM-CNN)           | Biomedical Signals (EMG)   | Automated biomedical signal generation addressing data scarcity in ALS       |
| Cognitive Decline     | Stasolla & Di Gioia (2023) | RL in Virtual Reality             | Cognitive Task Performance | Proposed personalized difficulty adjustment for cognitive training           |
| Motor Rehabilitation  | Pelosi et al. (2024)       | Q-learning in VR Environment      | Kinematic Movement Data    | Adaptive rehabilitation improved patient motor recovery outcomes             |


---

### **3.1 Diffusion Models and Neurological Disorders**

Diffusion models have garnered substantial attention in the domain of brain image synthesis and data augmentation, particularly due to their ability to generate high-resolution, realistic 3D images.

One of the most comprehensive studies in this domain is by Pinaya et al. (2022), who developed a latent diffusion model capable of generating synthetic 3D brain MRI images from large-scale datasets. Using approximately 31,740 T1-weighted brain MRIs from the UK Biobank, their model generated over 100,000 synthetic MRIs conditioned on variables such as age, sex, and brain structure volumes. The public release of this synthetic dataset has provided a valuable resource for the broader research community, addressing the critical issue of data scarcity in medical imaging.

Another important contribution was made by Peng et al. (2023), who proposed a conditional diffusion probabilistic model for realistic brain MRI generation. Their approach focuses on generating MRI subvolumes with anatomical consistency using slice-to-slice attention networks. This methodology is particularly advantageous in terms of memory efficiency, as it allows high-quality 3D image reconstruction without requiring extensive GPU resources.

While diffusion models have shown promise in anomaly detection for medical imaging, current applications primarily focus on structural abnormalities in brain MRI. These models can be trained on normative (healthy) brain data to identify pathological deviations, offering potential applications in detecting white matter hyperintensities, multiple sclerosis lesions, and brain tumors.

In longitudinal studies of disorders like Alzheimer's disease, complete temporal MRI datasets are often unavailable due to subject dropout or technical issues. To address this, Yuan et al. (2024) proposed a 3D diffusion-based model named "ReMiND" that imputes missing MRI volumes by conditioning on available prior (and optionally future) scans. Evaluated on the ADNI dataset, ReMiND outperformed forward-filling and variational autoencoder (VAE)-based methods in both imputation error and prediction of brain atrophy patterns, especially in the hippocampus.

Multiple sclerosis research has benefited significantly from diffusion models and GAN-based approaches for lesion analysis and MRI synthesis. Valencia et al. (2022) developed a conditional 3D GAN using a Pix2Pix architecture for T1-weighted image synthesis from T2-FLAIR sequences. Their approach achieved improved sensitivity for new T2 lesion detection, directly addressing data scarcity issues in MS imaging protocols. The methodology was validated on clinical datasets combining T2-FLAIR and T1-weighted images, demonstrating practical relevance for standardizing MS imaging workflows.

Salem et al. (2020) introduced a U-Net encoder-decoder architecture for MS lesion detection, achieving an 83% true positive rate with a 9.36% false positive rate using multi-modal MRI data (T1-w, T2-w, PD-w, FLAIR) from 60 patients. This work demonstrated how synthetic lesion generation can enhance training datasets for deep learning models in MS diagnosis.

Recent advances include the Brain Latent Progression (BrLP) model from Puglisi et al. (2024), which represents the most sophisticated application of diffusion models to neurological disease progression. Using latent diffusion models with ControlNet on 11,730 T1-weighted brain MRIs from 2,805 subjects, they achieved a 22% increase in volumetric accuracy and 43% improvement in image similarity for individual-level disease progression prediction.

Current research in EEG-based neurological disorder analysis primarily relies on established deep learning approaches rather than diffusion models. Recent advances include multi-feature fusion networks for Alzheimer's disease detection and graph convolutional neural networks for epilepsy prediction.

Chen et al. (2023) developed a multi-feature fusion learning approach for Alzheimer's disease prediction using resting-state EEG signals. Their method integrates spectral, temporal, and spatial features to achieve improved diagnostic accuracy. Similarly, Zheng et al. (2023) proposed an integrated approach combining spectrum, complexity, and synchronization signal features for Alzheimer's diagnosis via resting-state EEG.

For epilepsy prediction, Kuang et al. (2024) implemented a sophisticated framework combining graph convolutional neural networks with long- and short-term memory cell networks. This approach achieved high accuracy in seizure prediction by capturing both spatial relationships between EEG channels and temporal dependencies in the signal. Liu et al. (2024) further contributed to the field by developing a pseudo-three-dimensional CNN approach for epileptic seizure prediction based on EEG signals.

---

### **3.2 Neural Ordinary Differential Equations in Neurological Modeling**

Neural ordinary differential equations (Neural ODEs) offer a principled way to model the continuous-time evolution of complex systems. Their formulation is particularly advantageous in neurological contexts, where brain dynamics and disease progression are inherently gradual and temporal in nature. Traditional models such as recurrent neural networks (RNNs) or Transformers, while powerful, struggle with irregular sampling and missing timepoints that are characteristic of clinical data. Neural ODEs overcome this limitation by learning differential equations that govern the latent dynamics of data in continuous time.

One of the most compelling uses of Neural ODEs in neuroscience is their application to simulating brain dynamics from functional MRI (fMRI) data. Kashyap et al. (2023) demonstrated how neural ODEs combined with LSTM networks can estimate initial conditions of Brain Network Models in reference to measured fMRI data. Their approach involved analyzing whole-brain dynamics across 407 subjects from the Human Connectome Project, allowing for a more accurate and dynamic representation of brain behavior. This work highlighted the utility of ODEs in modeling large-scale brain networks with subject-specific dynamics.

While Neural ODEs show theoretical promise for disease progression modeling, current clinical applications remain limited. The field instead relies more heavily on traditional ODE approaches and advanced graph neural networks. Lian et al. (2024) developed a novel multi-modal graph approach for personalized progression modelling and prediction in Parkinson's disease, achieving superior performance over conventional methods. This approach integrates multiple data modalities to capture individual disease trajectories.

For Alzheimer's disease progression, Bossa & Sahli (2023) employed a multidimensional ODE-based model that captures disease dynamics using conventional differential equation frameworks rather than neural ODEs. Their model demonstrates how mathematical modeling can provide insights into disease progression patterns.

A significant advancement in MS research comes from Qian et al. (2021), who developed Latent Hybridisation Models (LHM) that combine expert-designed ODEs with Neural ODEs for disease progression modeling. Published at NeurIPS 2021, their framework integrates domain knowledge with data-driven approaches, outperforming baseline methods especially with limited training data. While initially demonstrated on COVID-19 intensive care data, the framework directly applies to MS disease progression and treatment optimization scenarios.

The integration of Neural ODEs with diffusion models shows particular promise for MS progression modeling, as demonstrated by the Brain Latent Progression (BrLP) model mentioned earlier, which combines multiple generative approaches for comprehensive disease trajectory prediction.

---

### **3.3. Reinforcement Learning Applications in Neurological Care**

Reinforcement learning (RL) has emerged as a powerful approach for developing adaptive, closed-loop decision-making systems in healthcare. In the field of neurology, RL has been applied to optimize therapeutic interventions such as deep brain stimulation (DBS), as well as to personalize rehabilitation and cognitive training.

In Parkinson's disease, DBS is widely used to alleviate motor symptoms such as tremor and rigidity. Traditionally, DBS systems are open-loop, meaning that they deliver fixed stimulation parameters regardless of real-time patient response. Cho et al. (2024) tackled this limitation by developing a closed-loop deep brain stimulation system with reinforcement learning and neural simulation. They compared several agents—including Soft Actor-Critic (SAC), Twin Delayed DDPG (TD3), Proximal Policy Optimization (PPO), and Advantage Actor-Critic (A2C)—to optimize stimulation protocols using basal ganglia-thalamic computational models. The TD3 agent achieved the best performance, identifying policies that consumed significantly less energy than conventional settings while preserving motor efficacy and reducing abnormal thalamic responses. Their results illustrate the feasibility of RL-based personalized neuromodulation systems.

Beyond neuromodulation, RL has also been integrated into motor rehabilitation systems. Pelosi et al. (2024) developed a personalized rehabilitation approach for reaching movement using reinforcement learning. Their VR therapy platform enables patients to perform reaching exercises by interacting with virtual objects, where a Q-learning agent adjusts the difficulty level in real time based on the patient's kinematic performance. This approach promotes engagement and progressive motor recovery, exemplifying how RL can deliver individualized, performance-sensitive rehabilitation protocols, especially in stroke or post-operative recovery settings.

Cognitive decline in conditions like Alzheimer's and mild neurocognitive disorder (MND) also presents opportunities for RL-driven intervention. Stasolla & Di Gioia (2023) explored the use of RL agents embedded within VR platforms to dynamically adjust the difficulty of cognitive tasks based on user behavior. Their perspective paper proposed tailored cognitive exercises that could enhance performance while improving user satisfaction and reducing caregiver burden. Such personalized digital therapies may become increasingly relevant in the early stages of dementia care.

Amyotrophic Lateral Sclerosis research has seen significant advances in reinforcement learning applications for speech synthesis and communication support. Regondi et al. (2025) published groundbreaking work in Scientific Reports demonstrating HiFi-GAN-based voice synthesis for personalized voice banking in ALS patients. Their system addresses progressive speech loss by generating high-quality synthetic voices with exceptional expressive and audio quality, representing direct clinical utility for patient communication support.

This work is complemented by the VOC-ALS Database established by Dubbioso et al. (2024), which analyzed 1,224 voice signals from 153 participants (51 controls, 102 ALS patients). Their F0 standard deviation analysis showed excellent ability to identify ALS and dysarthria severity, providing quantitative biomarkers for disease monitoring that can be integrated with RL-based speech synthesis systems.

Sengur et al. (2017) conducted pioneering work combining DCGAN for EMG signal processing with reinforcement learning strategies, achieving 96.80% classification accuracy for ALS vs. control classification using EMG signals from 89 ALS patients and 133 controls. This represents the first reported use of reinforcement learning in ALS EMG analysis, demonstrating feasibility of synthetic EMG data generation for diagnostic applications.

Building on this foundation, Hazra and Byun (2021) developed SynSigGAN using bidirectional grid LSTM generators with CNN discriminators for automated biomedical signal generation including EMG, addressing data scarcity issues critical for ALS research given limited patient populations.

---

### **3.4 Challenges Ethical Considerations and Future Directions**

The reviewed methodologies—diffusion models, neural ODEs, and reinforcement learning—differ fundamentally in their mechanisms and application contexts. Diffusion models have proven most effective in data augmentation and image reconstruction, especially where training data are limited or missing. Their capacity to synthesize high-resolution MRIs has significant implications for diagnostic imaging pipelines.

In contrast, Neural ODEs are uniquely suited to tasks involving brain dynamics modeling, though their application to disease progression prediction remains limited in practice. Traditional ODE approaches and graph neural networks currently dominate disease progression modeling applications.

Reinforcement learning excels in dynamic, feedback-sensitive settings. Its real-time learning and policy optimization capabilities make it ideal for closed-loop systems such as DBS controllers, rehabilitation programs, and adaptive cognitive training.

While all three approaches offer substantial benefits, their successful application depends on the task at hand. Furthermore, the models differ in terms of computational requirements, interpretability, and integration into clinical workflows. For instance, diffusion models and RL agents often suffer from "black box" opacity, raising concerns about trust and accountability in medical decision-making.

Despite their promise, the deployment of these advanced models in clinical neurology is fraught with challenges. One of the most critical is data scarcity. Medical imaging and electrophysiological datasets are expensive and time-consuming to collect, and privacy regulations often hinder data sharing. Even diffusion models, which are often touted as a remedy for data scarcity, require large volumes of high-quality training data to avoid overfitting and memorization. Several studies have cautioned that diffusion models can inadvertently replicate training samples, posing potential privacy risks.

Interpretability remains a persistent concern. Clinicians are understandably hesitant to rely on models whose decisions cannot be explained in human-interpretable terms. This issue is particularly acute in RL systems, which learn policies through trial-and-error exploration and are inherently difficult to audit.

Bias and fairness also demand attention. If training datasets reflect demographic imbalances—such as underrepresentation of minority populations—then generative models may perpetuate or even exacerbate these biases in their outputs. Ethical deployment of such technologies requires rigorous validation, fairness audits, and transparency in both development and application phases.

This chapter reviewed recent applications of deep learning-based generative models—specifically diffusion models, neural ODEs, and reinforcement learning—in the context of neurological disorders. Each technique contributes uniquely to overcoming the limitations of traditional methods. Diffusion models excel in medical image synthesis and data augmentation, Neural ODEs provide frameworks for modeling brain dynamics, and reinforcement learning enables real-time, adaptive intervention design.

However, challenges remain. Data quality, interpretability, ethical safety, and fairness are all crucial for real-world integration. Current research shows that while diffusion models have achieved significant success in brain MRI applications, their extension to EEG and other neurophysiological signals remains largely unexplored. Neural ODEs show promise for brain dynamics modeling but have limited practical applications in disease progression prediction. Reinforcement learning demonstrates the strongest clinical validation across multiple neurological applications.

Future studies should focus on hybrid models that combine the strengths of these approaches, the development of explainable AI frameworks for clinical applications, and the integration of multimodal datasets (e.g., MRI + PET + EEG). As the field evolves, these methods are likely to play a transformative role in patient-specific diagnosis and treatment planning.

---

## References

* Akbar, M. U., Larsson, M., Blystad, I., & Eklund, A. (2024). **Brain tumor segmentation using synthetic MR images – A comparison of GANs and diffusion models.** *Scientific Data, 11*, 259. [https://doi.org/10.1038/s41597-024-03073-x](https://doi.org/10.1038/s41597-024-03073-x)
* Bossa, M. N., Nakshathri, A. G., Díaz Berenguer, A., & Sahli, H. (2024). **Generative AI unlocks PET insights: Brain amyloid dynamics and quantification.** *Frontiers in Aging Neuroscience, 16*, 1410844. [https://doi.org/10.3389/fnagi.2024.1410844](https://doi.org/10.3389/fnagi.2024.1410844)
* Cai, J., Zhu, H., Liu, S., Qi, Y., & Chen, R. (2024). **Lung image segmentation via generative adversarial networks.** *Frontiers in Physiology, 15*, 1408832. [https://doi.org/10.3389/fphys.2024.1408832](https://doi.org/10.3389/fphys.2024.1408832)
* Costa de Farias, E., di Noia, C., Han, C., Sala, E., & Castelli, M. (2021). **Impact of GAN-based lesion-focused medical image super-resolution on the robustness of radiomic features.** *Scientific Reports, 11*, 21361. [https://doi.org/10.1038/s41598-021-00898-z](https://doi.org/10.1038/s41598-021-00898-z)
* Dubbioso, R., Pellegrino, G., Antenora, A., et al. (2024). **Voice signals database of ALS patients with different dysarthria severity and healthy controls.** *Scientific Data*, 11, 597. https://doi.org/10.1038/s41597-024-03597-2
* Gulakala, R., Markert, B., & Stoffel, M. (2022). **Generative adversarial network based data augmentation for CNN based detection of COVID-19.** *Scientific Reports, 12*, 19186. [https://doi.org/10.1038/s41598-022-23692-x](https://doi.org/10.1038/s41598-022-23692-x)
* Hazra, A., & Byun, Y. C. (2021). **SynSigGAN: Generative adversarial networks for synthetic biomedical signal generation.** *Biology*, 9(12), 441. https://doi.org/10.3390/biology9120441
* Puglisi, G., Ribeiro, A. H., Lorenzi, M., et al. (2024). **Enhancing spatiotemporal disease progression models via latent diffusion and prior knowledge.** In *Medical Image Computing and Computer Assisted Intervention – MICCAI 2024* (pp. 178-188). Springer. https://doi.org/10.1007/978-3-031-72069-7_17
* Qian, Z., Zame, W. R., Fleuren, L. M., et al. (2021). **Integrating expert ODEs into neural ODEs: Pharmacology and disease progression.** *Advances in Neural Information Processing Systems*, 34, 15833-15845. https://proceedings.neurips.cc/paper/2021/hash/5ea1649a31336092c05438df996a3e59-Abstract.html
* Regondi, S., Celardo, A., Pugliese, R., et al. (2025). **Artificial intelligence empowered voice generation for amyotrophic lateral sclerosis patients.** *Scientific Reports*, 15, 1247. https://doi.org/10.1038/s41598-024-84728-y
* Salem, M., Cabezas, M., Valverde, S., et al. (2020). **A fully convolutional neural network for new T2-w lesion detection in multiple sclerosis.** *NeuroImage: Clinical*, 25, 102149. https://doi.org/10.1016/j.nicl.2019.102149
* Sengur, A., Akbulut, Y., Guo, Y., & Bajaj, V. (2017). **Classification of amyotrophic lateral sclerosis disease based on convolutional neural network and reinforcement sample learning algorithm.** *Health Information Science and Systems*, 5, 9. https://doi.org/10.1007/s13755-017-0033-4
* Valencia, L. M., Dyrby, T. B., Lunau Fernandez, M., et al. (2022). **Evaluating the use of synthetic T1-w images in new T2 lesion detection in multiple sclerosis.** *Frontiers in Neuroscience*, 16, 954662. https://doi.org/10.3389/fnins.2022.954662
* Zhang, J., He, X., Qing, L., Gao, F., & Wang, B. (2022). **BPGAN: Brain PET synthesis from MRI using generative adversarial network for multi-modal Alzheimer's disease diagnosis.** *Computer Methods and Programs in Biomedicine, 217*, 106676. [https://doi.org/10.1016/j.cmpb.2022.106676](https://doi.org/10.1016/j.cmpb.2022.106676)
* Yuda, E., Ando, T., Kaneko, I., Yoshida, Y., & Hirahara, D. (2024). **Comprehensive Data Augmentation Approach Using WGAN-GP and UMAP for Enhancing Alzheimer's Disease Diagnosis.** *Electronics, 13*(18), 3671. [https://doi.org/10.3390/electronics13183671](https://doi.org/10.3390/electronics13183671)
* Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). **Unpaired image-to-image translation using cycle-consistent adversarial networks.** *Proceedings of the IEEE International Conference on Computer Vision*, 2223-2232. [https://doi.org/10.1109/ICCV.2017.244](https://doi.org/10.1109/ICCV.2017.244)

* Han, K., Wen, H., Shi, J., Lu, K.-H., Zhang, Y., Di, F., & Liu, Z. (2019). Variational autoencoder: An unsupervised model for encoding and decoding fMRI activity in visual cortex. *NeuroImage*, 198, 125–136. https://doi.org/10.1016/j.neuroimage.2019.05.039

* Hassanaly, R., Brianceau, C., Colliot, O., & Burgos, N. (2024). Unsupervised Anomaly Detection in 3D Brain FDG-PET: A Benchmark of 17 VAE-Based Approaches. In *Deep Generative Models (MICCAI 2023), LNCS 14533*, 110–120. https://doi.org/10.1007/978-3-031-53767-7_11
  
* He, R., Ang, G., Tward, D., for the Alzheimer's Disease Neuroimaging Initiative. (2025). Individualized Multi-horizon MRI Trajectory Prediction for Alzheimer's Disease. In: Schroder, A., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024 Workshops. MICCAI 2024. Lecture Notes in Computer Science, vol 15401. Springer, Cham. https://doi.org/10.1007/978-3-031-84525-3_3

* Kim, J.-H., Zhang, Y., Han, K., Wen, Z., Choi, M., & Liu, Z. (2021). Representation learning of resting state fMRI with variational autoencoder. *NeuroImage*, 241, 118423. https://doi.org/10.1016/j.neuroimage.2021.118423

* Kumar, S., Payne, P., & Sotiras, A. (2022). Normative Modeling using Multimodal Variational Autoencoders to Identify Abnormal Brain Structural Patterns in Alzheimer Disease. Proc SPIE Int Soc Opt Eng. 2023 Feb;12465:1246503. https://doi.org/10.1117/12.2654369. Epub 2023 Apr 7. PMID: 38130873; PMCID: PMC10731988.

* Yue, Y., De Ridder, D., Manning, P., Deng, J.D. (2025). Variational Autoencoder Learns Better Feature Representations for EEG-Based Obesity Classification. In: Antonacopoulos, A., Chaudhuri, S., Chellappa, R., Liu, CL., Bhattacharya, S., Pal, U. (eds) Pattern Recognition. ICPR 2024. Lecture Notes in Computer Science, vol 15323. Springer, Cham. https://doi.org/10.1007/978-3-031-78347-0_12

* Bossa, M. N., & Sahli, H. (2023). A multidimensional ODE-based model of Alzheimer's disease progression. *Scientific Reports*, 13, 3162. https://doi.org/10.1038/s41598-023-29383-5

* Chen, Y., Wang, H., Zhang, D., Zhang, L., & Tao, L. (2023). Multi-feature fusion learning for Alzheimer's disease prediction using EEG signals in resting state. *Frontiers in Neuroscience*, 17, 1272834. https://doi.org/10.3389/fnins.2023.1272834

* Cho, C. H., Huang, P. J., Chen, M. C., & Lin, C. W. (2024). Closed-loop deep brain stimulation with reinforcement learning and neural simulation. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 32, 3615-3624. https://doi.org/10.1109/TNSRE.2024.3465243

* Kashyap, S., Sanz-Leon, P., & Breakspear, M. (2023). A deep learning approach to estimating initial conditions of brain network models in reference to measured fMRI data. *Frontiers in Neuroscience*, 17, 1159914. https://doi.org/10.3389/fnins.2023.1159914

* Kuang, Z., Liu, S., Zhao, J., Wang, L., & Li, Y. (2024). Epilepsy EEG seizure prediction based on the combination of graph convolutional neural network combined with long- and short-term memory cell network. *Applied Sciences*, 14(24), 11569. https://doi.org/10.3390/app142411569

* Lian, J., Luo, X., Wang, H., Chen, L., Ge, B., Wu, F. X., & Wang, J. (2024). Personalized progression modelling and prediction in Parkinson's disease with a novel multi-modal graph approach. *npj Parkinson's Disease*, 10, 229. https://doi.org/10.1038/s41531-024-00832-w

* Liu, X., Li, C., Lou, X., Wang, L., & Chen, L. (2024). Epileptic seizure prediction based on EEG using pseudo-three-dimensional CNN. *Frontiers in Neuroinformatics*, 18, 1354436. https://doi.org/10.3389/fninf.2024.1354436

* Pelosi, A. D., Roth, N., Yehoshua, T., Tepper, M., Ashkenazy, Y., & Hausdorff, J. M. (2024). Personalized rehabilitation approach for reaching movement using reinforcement learning. *Scientific Reports*, 14, 17675. https://doi.org/10.1038/s41598-024-64514-6

* Peng, H., Gong, W., Beckmann, C. F., Vedaldi, A., & Smith, S. M. (2023). Accurate brain age prediction with lightweight deep neural networks. *Medical Image Analysis*, 68, 101871. https://doi.org/10.1016/j.media.2020.101871

* Pinaya, W. H. L., Tudosiu, P. D., Dafflon, J., Da Costa, P. F., Fernandez, V., Nachev, P., ... & Cardoso, M. J. (2022). Brain imaging generation with latent diffusion models. In *Deep Generative Models: Second MICCAI Workshop, DGM4MICCAI 2022* (pp. 117-126). Springer. https://doi.org/10.1007/978-3-031-18576-2_12

* Stasolla, F., & Di Gioia, C. (2023). Combining reinforcement learning and virtual reality in mild neurocognitive impairment: A new usability assessment on patients and caregivers. *Frontiers in Aging Neuroscience*, 15, 1189498. https://doi.org/10.3389/fnagi.2023.1189498

* Yin, W., Zhu, W., Gao, H., Zhao, H., Zhang, T., Zhang, C., ... & Hu, B. (2024). Gait analysis in the early stage of Parkinson's disease with a machine learning approach. *Frontiers in Neurology*, 15, 1472956. https://doi.org/10.3389/fneur.2024.1472956

* Yuan, C., Duan, J., Xu, K., Tustison, N. J., Hubbard, R. A., & Linn, K. A. (2024). ReMiND: Recovery of missing neuroimaging using diffusion models with application to Alzheimer's disease. *Imaging Neuroscience*, 2, 1-14. https://doi.org/10.1162/imag_a_00323

* Zheng, X., Wang, B., Liu, H., Sun, H., Li, M., Chen, W., & Zhang, L. (2023). Diagnosis of Alzheimer's disease via resting-state EEG: Integration of spectrum, complexity, and synchronization signal features. *Frontiers in Aging Neuroscience*, 15, 1288295. https://doi.org/10.3389/fnagi.2023.1288295
