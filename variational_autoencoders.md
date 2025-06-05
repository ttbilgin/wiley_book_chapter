# Variational Autoencoders in Medical Imaging

## 1. Basics of VAEs

A Variational Autoencoder (VAE) is a generative deep model combining an **encoder-decoder** architecture with a probabilistic latent space. The encoder maps each input (e.g. an image **x**) to parameters of a latent distribution $$q(z|x)$$ (typically Gaussian with mean μ(x) and variance σ²(x)). A latent vector **z** is sampled from this distribution and fed to the decoder, which reconstructs the input. Training maximizes the *evidence lower bound* (ELBO), comprising a reconstruction loss plus a Kullback–Leibler (KL) divergence between $$q(z|x)$$ and a chosen prior $$p(z)$$ (usually $$N(0,I)$$). The KL term regularizes the latent space and prevents overfitting. In β-VAEs, a weight β scales the KL term, trading off reconstruction fidelity against latent smoothness. After training, sampling z from the prior and decoding produces new synthetic images that resemble real data. In summary, VAEs learn a compressed **latent representation** of data (often lower-dimensional), while enabling generation of new samples via the probabilistic latent space.

Key components include:

* **Encoder:** Neural network producing latent parameters ($$\mu(x)$$, $$\sigma(x)$$) from input. 
* **Latent space:** Continuous probabilistic embedding; in a standard VAE each latent dimension is Gaussian.
* **Decoder:** Neural network that maps a sampled z back to the data space to reconstruct $$\hat x$$.
* **Loss (ELBO):** $$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \mathrm{KL}[q(z|x)\|p(z)]$$. 

For implementation, this is often approximated as: $$\mathcal{L} \approx \frac{1}{2\sigma^2}\|x - \hat x\|^2 + \beta\,\mathrm{KL}(q(z|x)\|N(0,I)) + \text{constants}$$, where the scaling factor $$\frac{1}{2\sigma^2}$$ depends on the assumed decoder output variance and β controls the KL regularization strength.

VAEs are widely used in medical imaging because they can augment datasets by generating realistic synthetic examples. Their probabilistic nature and smooth latent space facilitate tasks like anomaly detection (by measuring likelihood under the model) and feature extraction.

## 2. Applications of VAEs in Neuroscience (2019–2024)

VAEs have been applied across neuroscience modalities (MRI, fMRI, PET, EEG) for diverse tasks (classification, progression modeling, synthesis, anomaly detection, dimensionality reduction). Recent studies include:

| Study (Year)              | Data Type (Modality)      | Task / Disease                                  | VAE Model                              | Key Findings                                                                                                                                                                                                                                                                                      |
| ------------------------- | ------------------------- | ----------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kumar *et al.* (2022)     | Multimodal MRI (T1, T2)   | Alzheimer's disease (normative modeling)        | Multimodal VAE                         | Proposed a **multi-modal VAE** on joint T1/T2 brain volumes to learn healthy controls' distribution. Testing on AD patients, the method produced deviation maps more sensitive to disease stage and better correlated with cognition than a unimodal baseline.                               |
| He *et al.* (2024)        | Longitudinal MRI (ADNI)   | Alzheimer's progression (trajectory prediction) | Conditional VAE (CVAE)                 | Used a CVAE to generate individualized future MRI scans given baseline scan, age, and diagnosis. The model could **extrapolate brain changes** and produced synthetic data useful for downstream tasks (anomaly detection, classification).                                                     |
| Hassanaly *et al.* (2024) | 3D FDG-PET                | Dementia (unsupervised anomaly detection)       | VAE ensemble (17 variants)             | Benchmarked 15 VAE variants (plus vanilla AE/VAE) for detecting anomalies in PET related to Alzheimer's. Nine variants yielded accurate reconstructions and healthy-looking images. **Many VAE variants generalized well**, but **none outperformed the basic VAE/AE** in anomaly detection. |
| Kim *et al.* (2021)       | Resting-state fMRI        | Individual identification (no disease)          | β-VAE                                  | Applied a β-VAE on large HCP rs-fMRI data. The **latent features** learned were highly informative: they could *reliably identify subjects* from their fMRI patterns with ~99% accuracy, demonstrating that the VAE captured individual brain signatures.                                     |
| Han *et al.* (2019)       | Task fMRI (visual cortex) | Visual encoding/decoding                        | Standard VAE                           | Trained a VAE to model V1 visual cortex responses to natural images. The encoder-decoder learned to **reconstruct stimulus images from fMRI** and vice versa, serving as an unsupervised model of the visual processing pathway.                                                             |
| Yue *et al.* (2023)       | EEG (resting state)       | Obesity vs. lean subject classification         | VAE feature extractor + CNN classifier | Introduced a VAE to extract subject-invariant EEG features for obesity classification. Using these latent features in a 1D-CNN classifier yielded **significantly higher accuracy** for distinguishing obese and lean groups, outperforming conventional classifiers.                        |

These examples illustrate key use-cases: VAEs are used to **extract low-dimensional representations** of brain data (for classification or clustering), to **synthesize or predict images** (augmenting or simulating longitudinal scans), and to **detect anomalies** by modeling normal anatomy. For instance, Kumar et al. used a multimodal VAE to capture complex structural changes across MRI modalities. VAE-based generators like the CVAE in He et al. can predict future brain scans, aiding disease modeling. In fMRI, unsupervised VAEs have been shown to uncover meaningful latent factors, such as individual-specific network features and visual cortex representations. Overall, VAEs have proven valuable in neuroscience for **dimensionality reduction** and **feature learning**, especially when probabilistic embeddings and reconstruction losses align with clinical data characteristics.

## 3. Comparing VAEs and GANs in Healthcare Research

Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) are two dominant deep generative frameworks in medical AI. Recent literature (2019–2024) highlights complementary strengths and weaknesses for healthcare tasks, though diffusion models are increasingly challenging both approaches:

* **Image Synthesis:** GANs typically produce *higher-fidelity, sharper images* by training a generator against a discriminator, but they can suffer from training instability and mode collapse. VAEs generate more *diverse* outputs (covering more modes) and are easy to train, but their reconstructions tend to be **blurry** due to the Gaussian prior and KL regularization. As noted in recent reviews, *"VAEs outperform GANs in terms of output diversity and are free of mode collapse. However… \[they have a] tendency to often produce blurry and hazy output images"*. Hybrid models (e.g. VAE-GAN) attempt to combine the VAE's latent smoothness with GAN's image sharpness. However, **diffusion models now often outperform both** in medical imaging applications, achieving better diversity and fidelity.

* **Data Imputation:** GAN-based methods like **GAIN** (Generative Adversarial Imputation Nets) have shown strong performance for imputing missing clinical data, especially at high missingness rates. VAE-based imputers (e.g. MIWAE, HI-VAE) can also handle mixed-type missing data and model uncertainty, but benchmarks indicate their accuracy may degrade when data are highly heterogeneous or missingness is extreme. In practice, GANs offer **fast, accurate completion** for EHR/time-series, whereas VAEs provide principled uncertainty but can require careful tuning.

* **Disease Progression Modeling:** VAEs (especially *conditional VAEs*) naturally model distributions and can generate plausible future observations. For example, VAEs have been used to simulate disease progression by sampling from learned trajectories. GANs can likewise generate longitudinal images (e.g. by conditioning on time), but require more complex adversarial setups. GANs may produce more realistic individual scans, but VAEs provide explicit likelihoods and smooth interpolation over disease stages.

* **Representation Learning & Interpretability:** VAEs yield an explicit latent space with a known prior, making them more interpretable: the latent dimensions capture meaningful features of the data. In contrast, GAN latent spaces lack an explicit probability model and are harder to interrogate. As a result, VAEs are often preferred when a low-dimensional embedding or feature attribution is needed (e.g. for biomarker discovery). 

| **Application/Task**               | **VAE (Advantages)**                                                                                     | **GAN (Advantages)**                                                                      | **Remarks**                                                                                                                                            |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Image Synthesis (Augmentation)** | - High sample diversity; no mode collapse<br/>- Stable, likelihood-based training                        | - Produces sharp, realistic images<br/>- Well-established in imaging (many architectures) | VAEs produce blurrier images; GANs may ignore some modes. Diffusion models often outperform both in recent medical imaging studies.                  |
| **Data Imputation (Missing Data)** | - Provides probabilistic estimates (uncertainty) for imputations                                         | - High accuracy in practice (e.g. GAIN)<br/>- Fast convergence                            | VAE-based imputers exist (MIWAE, HI-VAE) but can falter with complex EHR patterns. GAN imputers handle heterogeneous mixes well.                       |
| **Disease Progression Modeling**   | - Can model distributions of trajectories; handles uncertainty smoothly (e.g. CVAE)                      | - Can generate photo-realistic future states with adversarial loss                        | VAEs yield probabilistic forecasts useful for long-term predictions; GANs yield sharper, potentially more realistic snapshots but need careful design. |
| **Representation Learning**        | - Learns a smooth latent space reflecting data variation<br/>- Latents are interpretable (feature-based) | - No explicit latent inference (latent code is adversarially constrained)                 | VAEs explicitly encode data to z∼p(z\|x), facilitating analysis; GANs do not encode input to a latent by default (except autoencoder variants).        |
| **Interpretability**               | - Latent variables have statistical meaning; disentanglement possible                                    | - Limited interpretability; latent space not semantically regularized                     | The smooth VAE latent allows tracing how factors change; GAN's adversarial training makes disentanglement difficult.                                   |

In summary, **VAEs** are favored when robust, diverse sampling and interpretable embeddings are important. **GANs** excel when generating the highest-quality images or realistic samples is critical. Recent reviews emphasize this complementary nature. For example, VAEs are praised for *"ease of training and good coverage of the data distribution"*, whereas GANs are celebrated for image *fidelity*. In healthcare, the choice often depends on the application: e.g. GANs are widely used for **image synthesis/augmentation** (MRI, histology, etc.), while VAEs underpin tasks like **anomaly detection**, **latent factor analysis**, and **probabilistic progression modeling**. However, **diffusion models are increasingly preferred** for medical image synthesis due to their superior quality-diversity trade-offs.

## Challenges and Future Directions

Despite their promising applications, VAEs in medical imaging face several important challenges. **Evaluation metrics** for medical applications often require domain-specific criteria beyond traditional reconstruction losses. **Data scarcity** in medical domains means VAEs may struggle to capture true population variability, especially for rare conditions. **Regulatory validation** for clinical deployment requires extensive testing and approval processes that can take years.

Future research should focus on **multimodal VAEs** that integrate different imaging modalities, **disentangled representations** for better interpretability, and **hybrid approaches** combining VAEs with diffusion models or other generative techniques. The field is also moving toward **conditional VAEs** that can incorporate clinical metadata for more targeted synthesis and **federated learning** approaches that can train on distributed medical datasets while preserving privacy.

## References

* Dong, W., Fong, D. Y. T., Yoon, J., Wan, E. Y. F., Bedford, L. E., Tang, E. H. M., & Lam, C. L. K. (2021). Generative adversarial networks for imputing missing data for big data clinical research. *BMC Medical Research Methodology*, 21, 78. https://doi.org/10.1186/s12874-021-01272-3

* Han, K., Wen, H., Shi, J., Lu, K.-H., Zhang, Y., Di, F., & Liu, Z. (2019). Variational autoencoder: An unsupervised model for encoding and decoding fMRI activity in visual cortex. *NeuroImage*, 198, 125–136. https://doi.org/10.1016/j.neuroimage.2019.05.039

* Hassanaly, R., Brianceau, C., Colliot, O., & Burgos, N. (2024). Unsupervised Anomaly Detection in 3D Brain FDG-PET: A Benchmark of 17 VAE-Based Approaches. In *Deep Generative Models (MICCAI 2023), LNCS 14533*, 110–120. https://doi.org/10.1007/978-3-031-53767-7_11

* He, R., Tward, D. J., Johnson, K. E., Aksman, L., Pati, S., Greig, E. H., & Wolk, D. A. (2024). Individualized Multi-Horizon MRI Trajectory Prediction for Alzheimer's Disease. *arXiv preprint*. https://arxiv.org/abs/2408.02018

* Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Sy2fzU9gl

* Kebaili, A., Lapuyade-Lahorgue, J., & Ruan, S. (2023). Deep Learning Approaches for Data Augmentation in Medical Imaging: A Review. *Journal of Imaging*, 9(4), 81. https://doi.org/10.3390/jimaging9040081

* Kim, J.-H., Zhang, Y., Han, K., Wen, Z., Choi, M., & Liu, Z. (2021). Representation learning of resting state fMRI with variational autoencoder. *NeuroImage*, 241, 118423. https://doi.org/10.1016/j.neuroimage.2021.118423

* Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. In *International Conference on Learning Representations*. https://arxiv.org/abs/1312.6114

* Kumar, S., Payne, P., & Sotiras, A. (2022). Normative Modeling using Multimodal Variational Autoencoders to Identify Abnormal Brain Structural Patterns in Alzheimer Disease. *arXiv preprint*. https://arxiv.org/abs/2110.04903

* Maity, S., Mandal, R. P., Bhattacharjee, S., & Chatterjee, S. (2022). Variational Autoencoder-Based Imbalanced Alzheimer Detection Using Brain MRI Images. In *Lecture Notes in Networks and Systems, Vol. 402* (pp. 533–544). Springer. https://doi.org/10.1007/978-981-19-1657-1_14

* Yoon, J., Jordon, J., & van der Schaar, M. (2018). GAIN: Missing data imputation using generative adversarial nets. In *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80:5689-5698. https://proceedings.mlr.press/v80/yoon18a.html

* Yue, Y., Deng, J. D., De Ridder, D., Manning, P., & Adhia, D. (2023). Variational Autoencoder Learns Better Feature Representations for EEG-based Obesity Classification. *arXiv preprint*. https://arxiv.org/abs/2302.00789
