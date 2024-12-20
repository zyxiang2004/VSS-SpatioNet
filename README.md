# VSS-SpatioNet

This project contains the code for **[VSS-SpatioNet: A Multiscale Feature Fusion Network for Multimodal Image Integrations]** (Author: **Zeyu Xiang**). The code has been validated in MATLAB and Python environments and can be used to reproduce the experimental results presented in the paper.

## File Structure

- **analysis**: Contains the primary data analysis code.
  - **main_all.m**: Open this file with MATLAB to directly reproduce the data and results presented in the paper.

- **causal-conv1d**: Implements causal convolution for 1D operations, used in models involving causal convolution in the experiments.

- **mamba**: Contains auxiliary tools or libraries related to this project.

- **pytorch_msssim**: Includes a PyTorch implementation of Multi-Scale Structural Similarity (MS-SSIM) module, suitable for image similarity computation in deep learning.

- **selective_scan**: Contains selective scanning modules for specific algorithms and data processing steps described in the paper.

- **SS2D_arch.py**: Defines the SS2D architecture, providing model structure for the deep learning tasks.

- **args_fusion.py**: Configures parameters for training or testing the fusion network.

- **checkpoint.py**: Handles model checkpointing, including saving and loading parameters for trained models.

- **net.py**: Defines the main network architecture used in this study.

- **test_21pairs_axial.py**: Implements the testing procedure for the 21 pairs of axial data as detailed in the paper.

- **train_fusionnet_axial.py**: Trains the fusion network, suitable for the axial data used in the experiments.

- **utils.py**: Contains utility functions supporting the primary functions of the codebase.

## Reproduction Instructions

To assist users in reproducing the MATLAB experimental results from this study, please follow the steps below:

1. **Prepare the MATLAB Environment**:
   - Ensure that MATLAB is installed and that the required toolboxes are available.

2. **Execution Steps**:
   - Open MATLAB and set the working directory to the `analysis` folder.
   - Run the `main_all.m` file. This file includes comprehensive data processing and analysis code, which will automatically produce output consistent with the data and results reported in the paper.

3. **Output Verification**:
   - Users can compare the generated outputs with the data in the paper to validate the accuracy of the experimental results.

## Usage Instructions

The Python section includes several modules supporting the experiments and model construction detailed in the paper. To use this code, it is recommended to follow these steps:

1. Configure the Python environment, ensuring that necessary dependencies are installed (dependencies can be identified through comments in the code).

2. Different Python files provide functions for experiment configuration, model definition, testing, and training. Users can run the relevant scripts based on their experimental needs.

3. Specific code and file descriptions are available within the code comments, or users can flexibly combine and utilize these modules according to their experimental requirements. This allows for model training, testing, performance enhancement, and cross-domain transfer, showcasing the potential of the VSS-SpatioNet architecture in various domains.

---

This work is intended for submission to the *Scientific Reports* journal, where Open Access publication will facilitate wider scientific dissemination. By making VSS-SpatioNet accessible, I hope this work can serve as a resource that sparks further research and applications. My confidence in VSS-SpatioNet lies not in its current form alone, but in the potential it holds for adaptation and growth across diverse fields of study. I look forward to seeing how the community might expand on this foundation to address both known and unforeseen challenges in multimodal integration. 😊
