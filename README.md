# SPLEEN-US

#Background

The SPLEEN-US proposal aims to create a large, open cine-loop dataset of spleen ultrasound images with expert-validated segmentation masks and metadata for precision neuromodulation. The vision is to enrol ~2,000 healthy subjects, acquire cine ultrasound of the spleen and apply automated segmentation algorithms followed by proofreading.

#Precision neuromodulation and the need for an annotated spleen dataset

Focused ultrasound neuromodulation of the spleen has emerged as a promising non-pharmacological therapy for inflammatory diseases. Delivering such therapies safely requires precise localization of the spleen while avoiding surrounding anatomy (e.g., ribs, lungs, pancreas) and accounting for respiratory motion. Abdominal organs, including the liver, kidneys, and spleen, move significantly during respiration. Therefore, a neuromodulation system must track the spleen in real time to maintain the acoustic focus on the target while sparing adjacent tissues.
The student teams developed a variety of tools that could help realize the pipeline of raw ultrasound spleen image processing, automated segmentation with deep neural networks, proofreading for ground-truth annotations.

##Team 1

###Potential use:
•	Provides baseline segmentation models and training scripts for cine-loop data. 
•	Supports multiple architectures, enabling comparison with more advanced models. 
•	Can be fine-tuned on SPLEEN-US images to generate initial masks for proofreading.

###Relevance to neuromodulation:
•	Enables rapid prototyping of segmentation models for spleen localization. 
•	Baseline UNets can be fine-tuned on SPLEEN-US data to produce masks used by the neuromodulation system for targeting. 
•	Integrated visualization helps inspect failure cases and improve algorithm robustness.


##Team 2

###Potential use:
•	A plug-and-play segmentation package for SPLEEN-US: prompts like “spleen” can be used to generate masks on cine loops with minimal user input. 
•	Fine-tuned SAM models can serve as strong baseline algorithms, likely outperforming traditional UNets when trained on SPLEEN-US data. 
•	The API allows cascaded training (UNet++ followed by SAM), which may improve generalization across subjects and imaging machines. 

###Relevance to neuromodulation:
•	Provides high-accuracy spleen masks that can be updated in real time. 
•	Robust to variation across patients and scanning conditions due to foundation-model training; it helps avoid surrounding structures. 
•	Interactive prompt-based interface allows quick re-initialization if the target moves outside the region during respiration.

##Team 3

###Potential use:
•	Augment SPLEEN-US training data to mitigate domain shifts and improve generalization. 
•	Provide synthetic cine loops for rare or difficult views, filling gaps in the dataset. 
•	Allow ablation studies comparing training on real, synthetic, or mixed data.

###Relevance to neuromodulation:
•	Augments the training set with diverse examples of spleen positions and shapes across respiratory cycles. 
•	Helps models generalize to out-of-distribution scenarios, reducing the risk of mis-targeting. 
•	Enables simulations of extreme motions or unusual anatomies for stress testing algorithms.

##Team 4

###Potential use:
•	Acts as the quality control interface for SPLEEN-US; experts can quickly refine automatically generated masks. 
•	The SPLEEN-US workflow requires automated segmentation followed by expert proofreading, and this tool provides exactly that functionality.
•	The proofreading tool allows clinicians to quickly correct automatically generated masks, ensuring that training data accurately delineate the spleen and exclude adjacent organs.

###Relevance to neuromodulation:
•	Facilitates rapid generation of high-quality spleen masks for training and evaluation. 
•	Supports iterative improvement of algorithms by providing accurate labels for difficult frames (e.g., during deep inspiration when the spleen is obscured). 

##Team 5

###Potential use:
•	Use the API and folder organization to package SPLEEN-US images, masks, and metadata in a uniform format, facilitating reproducibility and ease of use. 
•	Combine existing public datasets with the new SPLEEN-US cine loops to train more general models. 
•	Provide data-loading utilities for researchers participating in SPLEEN-US challenges.

###Relevance to neuromodulation:
•	Provides a standard format and API for distributing SPLEEN-US data to researchers developing spleen-localization algorithms. 
•	Facilitates combining SPLEEN-US with other datasets to improve model generalization. 
•	Encourages reproducibility by ensuring that everyone trains on the same data splits and evaluation protocols.

In conclusion, the student solutions provide a strong starting point for building a common SPLEEN-US codebase for the project “Open Annotated Spleen Ultrasound Dataset for Precision Neuro-modulation.” The most important outcome of this review is that the five teams should not be treated as separate projects. Instead, their work can be reorganized into a unified pipeline for data preparation, spleen segmentation, expert proofreading, synthetic augmentation, and future motion-aware neuromodulation guidance.
The highest priority should be given to Team 5, because dataset preprocessing and standardization are the foundation of the entire SPLEEN-US project. Since SPLEEN-US aims to release approximately 2000 cine loops with masks, metadata, and public access, a clean and reproducible data format is essential before any model training or annotation workflow can be reliable.
After the dataset infrastructure, Team 1 and Team 2 should be integrated as the main deep-learning segmentation components. Team 1’s UNet-based approach is valuable as a conventional and interpretable baseline for spleen mask segmentation. It can be used to benchmark performance and provide an initial automatic mask for each frame. Team 2’s SAM/MedSAM-based approach should be considered the more advanced segmentation component because prompt-based and fine-tuned foundation models may improve spleen localization and generalization across different ultrasound appearances. Together, Team 1 and Team 2 can provide a strong segmentation module for identifying the spleen boundary, which is directly relevant to precision neuromodulation because accurate spleen localization is the first requirement for focused ultrasound targeting.
Team 4’s proofreading web application should be integrated after the automatic segmentation module. This is important because SPLEEN-US is not only a model-development project; it is also an annotated dataset project. Automatically generated masks will need expert correction before they can be released as high-quality ground truth. Team 4’s proofreading tool can therefore serve as the human-in-the-loop quality-control interface. This component is especially important for cine ultrasound, where respiratory motion may cause the spleen boundary to shift from frame to frame. Expert proofreading will help ensure that the final masks are accurate enough for training future localization and tracking algorithms.
Team 3’s synthetic data generation should be included as a later-stage augmentation and robustness module. Synthetic ultrasound images can help increase diversity in spleen shape, image quality, scanner appearance, and acoustic artifacts. This is useful because the SPLEEN-US summary already shows that a model trained on one dataset may fail on another dataset, indicating a lack of generalization. It should be used to support training, stress-test models, and improve robustness after the real-data pipeline is established.



