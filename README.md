
>📋  Accompanying code for "A U-Net model for Local Brain Age"

# A U-Net model for Local Brain Age

This repository is the official implementation of [Matrix Inversion free variational inference in Conditional Student's T Processes](https://openreview.net/pdf?id=jLLR71k9Hsi). 

>📋  ![Inverse Free Conditional Student-T Processes](inverse_free_student_t_processes.png)

## Requirements

To install requirements:

```setup
pip install tensorflow-gpu
```

## Training

To train the model in the paper, run this command:

```train
python3 full_training_script.py --num_encoding_layers=2 --num_filters=64 --num_subjects=2 --num_voxels_per_subject=2 --location_metadata=$absolute_path_of_metadata_for_dataset --dirpath_gm=$absolute_path_of_directory_of_spm12_processed_gray_matter_nifti_files
--dirpath_wm=$absolute_path_of_directory_of_spm12_processed_white_matter_nifti_files --dataset_name=$dataset_name
```

>📋  The above command line function start training the U-Net model provided a dataset that already went through spm12's Dartel pipeline. One can get the folder with gm and wm nifti files by using  batched_spm12_dartel(img_dir, name_of_dataset, size_batch) function inside dartel_pipeline.py. We mention that the training process is quite lengthy (around 3 weeks on a GPU) if one wants the best possible performance.

## Evaluation

To evaluate our already trained U-Net model on your dataset you need run the following command (if python throws error such as too many files opened, just type ulimit -n 10000 on the command line)

```eval
ython3 full_testing_script.py --num_encoding_layers=2 --num_filters=64 --num_subjects=2 --num_voxels_per_subject=2 --location_metadata=$absolute_path_of_metadata_for_dataset --dirpath_gm=$absolute_path_of_directory_of_spm12_processed_gray_matter_nifti_files
--dirpath_wm=$absolute_path_of_directory_of_spm12_processed_white_matter_nifti_files --dataset_name=$dataset_name
```




## Citing this work

TBD







