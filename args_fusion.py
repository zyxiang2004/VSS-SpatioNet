class args():

    # Training arguments
    epochs = <number_of_epochs>  # Placeholder for the number of training epochs
    batch_size = <batch_size_value>  # Placeholder for batch size value
    dataset_ir = "<path_to_infrared_dataset>"  # Placeholder for infrared dataset path
    dataset_vi = "<path_to_visible_dataset>"  # Placeholder for visible dataset path

    dataset = "<dataset_type>"  # Placeholder for dataset type

    HEIGHT = <image_height>  # Placeholder for image height
    WIDTH = <image_width>  # Placeholder for image width

    save_fusion_model = "<path_to_save_fusion_model>"  # Placeholder for directory to save fusion model
    save_loss_dir = "<path_to_save_loss_directory>"  # Placeholder for directory to save training loss

    image_size = <image_size_value>  # Placeholder for training image size
    cuda = <cuda_flag>  # Placeholder for CUDA flag, set to 1 for GPU, 0 for CPU
    seed = <random_seed>  # Placeholder for random seed value

    lr = <learning_rate>  # Placeholder for learning rate value
    log_interval = <log_interval_value>  # Placeholder for log interval for training loss
    resume_fusion_model = "<path_to_resume_fusion_model>"  # Placeholder for resuming fusion model path
    # nest net model
    resume_nestfuse = "<path_to_nestfuse_model>"  # Placeholder for pre-trained NestFuse model path
    resume_vit = "<path_to_vit_model>"  # Placeholder for Vision Transformer model path
    fusion_model = "<path_to_fusion_model>"  # Placeholder for fusion model path if any

    mode = "<operation_mode>"  # Placeholder for mode of operation
