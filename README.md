# FISHID 
    JOSEPH GOLDBLATT
    DEPARTMENT OF COMPUTER SCIENCE
    UNIVERSITY OF CAPE TOWN
    AUGUST 2025

    INVESTIGATING CONTRASTIVE LEARNING FOR FISH IDENTIFICATION AND CLASSIFICATION


## ABSTRACT
   This paper investigates the application of contrastive learning, a self-supervised machine learning technique, to the task of classifying fish species from underwater video. Specifically, we investigate classification of fish species from a custom dataset of underwater videos collected off the East Coast of South Africa by the South African Institute for Aquatic Biodiversity (SAIAB). This dataset contains hundreds of hours of unlabelled underwater footage, with sparse species annotations. The dataset is well suited to testing the application of contrastive learning, which has been shown to produce high classification accuracy when trained on a large unlabelled dataset and a small labelled one. We therefore implemented the SimCLR contrastive learning framework, first proposed by Google in 2020. We trained a ResNet-50 model using a large unlabelled dataset extracted from the SAIAB footage, supplemented with supervised training on a small annotated dataset. The model achieved a mean classification accuracy (across ten random seeds) of 70.65% on the labelled dataset of 38 classes. However, this accuracy was matched by a state of the art YOLOv11-n classifier model, suggesting that although there is scope for the application of self-supervised learning to this task, out-performing current supervised classifiers may require implementing a more modern self-supervised framework.

   The full write-up is included in this repository: Contrastive_Learning_for_Fish_ID.pdf


## PROJECT ENVIRONMENT 
   Run the command "pip install -r requirements.txt" to install all requirements necessary to run scripts from the following directories: labelled_data_preprocessing, unlabelled_data_preprocessing, yolo_testing, and results_and_analysis. (Note that running yolo_testing/yolo_trainer.py will require GPU access).

   To run the files contained in the directory simclr_train_and_eval, it is necessary to first create a Conda environment. The underlying package used for SimCLR training relies on now deprecated version of Python and PyTorch and can thus only be run from a virtual environment. The parameters for this environment are contained in the file "simclr_train_and_eval/environment.yml"

   To create this environment locally, from within the simclr_train_and_eval, run the command "conda env create -f environment.yml". To activate the environment, run the command "conda activate pytorch_env".

   On a SLURM scheduler based HPC, to create the environment, start an interactive job and run the following commands:
    "module load python/miniconda3-py3.12-usr"
    "conda env create -f environment.yml"
    To activate the environment run the command "source activate pytorch_env"


## PROJECT STRUCTURE
    The project is split into five sub-directories, each corresponding to a specific portion of the experimentation pipeline:

### 1 unlabelled_data_preprocessing
    This directory contains all files necessary for producing the formatted unlabelled dataset to be used for SimCLR training. 

#### unlabelled_cutout_extractor.py:
        Usage:
            python3 unlabelled_cutout_extractor.py <root_directory> <interval> <model_path> [confidence]
        Functionality:        
            - extracts videos from the SAIAB dataset.         
            - extracts frames at one second intervals from these videos (excluding the first ten and last five minutes, when the camera is typically above-surface).
            - Runs a pre-trained YOLOv11-n object detector (stored as detector/fish.pt) to predict bounding boxes corresponding to fish within these frames.
            - Extracts the predicted bounding boxes and stores each bounding box as a standalone "cutout".
            - Stores these cutouts in a directory titled "extracted_unlabelled_frames".

#### unlabelled_dataset_formatter.py:
        Usage:
            python3 unlabelled_dataset_formatter.py <source_dir> <output_dir> [train_ratio]
        Functionality:
            - Splits images in directory of unlabelled images into separate train and validation sub-directories, using a species ratio.


### 2 labelled_data_preprocessing
    This directory contains all files necessary for producing the formatted labelled dataset to be used for SimCLR evaluation.

#### dataset_extractor_filterer.py:
        Usage:
            python3 dataset_extractor_filterer.py <input_dir> <output_dir> <csv_mapping> [--min-images N]
        Functionality:
            - given a set of directories containing frames with corresponding bounding box annotation files, extracts cutouts corresponding to all bounding boxes with a class label >0.
            - stores these cutouts in a new directory, within class separated subdirectories. 
            - removes all sub-directories containing <3 cutouts.
            - visualizes the dataset as a pie chart, before and after filtering based on class-size.

#### dataset_splitter.py:
        Usage:
            python3 dataset_splitter.py <source_dir> <target_dir> [--min-per-split N]
        Functionality:
            - splits a class separated directory into train, test, and val directories, using a split ratio of 70/15/15. 
            - Ensures that a specified minimum number of samples from any class are present in all splits. 


### 3 yolo_testing:
    This directory contains the necessary script for testing the performance of a supervised YOLOv11-n-cls model trained on the labelled dataset.

#### yolo_trainer.py:
        Usage:
            python3 yolo_trainer.py --dataset-dir <dir> --output-dir <dir>   
        Functionality:
            - trains a YOLOv11-n-cls model using a labelled dataset stored in ImageNet format. 
            - performs training multiple times over a range of random seeds, storing detailed performance metrics for each training run. 


### 4 simclr_train_and_eval:  
    This directory contains the Python scripts necessary for training a ResNet model using the SimCLR framework, evaluating this model using a labelled dataset and evaluating a baseline ResNet model on the same dataset for comparison.
    Other than the script batch_evaluator.py, the entire contents of the sub-directory "simclr-pytorch" have been cloned from the open-source github repository accessible here: https://github.com/AndrewAtanov/simclr-pytorch. 

    In addition to these Python scripts, six bash scripts are included, these scripts are designed to orchestrate training and evaluation either locally or on a SLURM based High Performance Computing Cluster.
    - Scripts ending in "_HPC.sh" are designed for submission to a SLURM scheduler, and should be run using the command: "sbatch [script_name].sh". 
    - Scripts ending in "_locally.sh" are designed to be run locally (assuming local access to a GPU), and should be run using the command: "bash [script_name].sh".

    The functionality of the scripts is as follows:
        - train_simclr_[].sh: these scripts orchestrate the self-supervised training of a ResNet-50 model using the SimCLR framework and an unlabelled dataset of images.
        - evaluate_simclr_[].sh: these scripts orchestrate the evaluation of a SimCLR trained ResNet encoder. Evaluation is performed ten times, across ten random seeds, for statistical robustness. 
        - evaluate_resnet_[].sh: this script orchestrates the linear evaluation of a supervised ImageNet-trained ResNet-50 encoder. Evaluation is performed ten times, across ten random seeds, for statistical robustness. 

#### resnet_evaluator.py:
        Usage:
            python resnet_evaluator.py --data-dir <dir> --output-dir <dir> --run-name <name>
        Functionality:
            - performs linear evaluation of an ImageNet trained Resnet-50 encoder on a labelled dataset
            - saves results to a CSV file.

#### simclr-pytorch:    
        The open-source PyTorch-based SimCLR implementation used for SimCLR training. The original project can be found here: https://github.com/AndrewAtanov/simclr-pytorch.            

#### simclr-pytorch/batch_evaluator.py:
        Usage:
            python batch_evaluator.py --pretrain_dir <dir> --dataset_path <path> --output_dir <dir> 
        Functionality:
            - given the path to a pre-trained ResNet-50 encoder and the directory to a labelled dataset, performs linear and non-linear evaluation and partial and full fine-tuning of the encoder on that dataset
            - outputs the results of this evaluation to various csv and json files.


### 5 results_and_analysis:
    This directory contains the python scripts necessary to evaluate the results of the experiments run in the rest of the project. The directories simclr_eval_results, yolo_eval_results, resnet_baseline_results and fish4k_eval_results contain the results of the experiments run across the other directories of the project (these raw result directories are excluded from GitHub for size; summarised data and plots are retained in the data and plots sub-directories). 

#### simclr_eval_data_extractor.py:
        Usage:
            python3 simclr_eval_data_extractor.py
        Functionality:
            - extracts simple, structured data from the results of the evaluation of a SimCLR-trained ResNet-50 model
            and visualizes this data as a bar chart.
            - outputs results to  csv files: experiment_means.csv, experiment_stds.csv and experiment_statistics.csv.    

#### simclr_run_analyser.py:
        Usage:
            python3 simclr_run_analyser.py
        Functionality:
            - extracts data for a single evaluation metric of a single SimCLR model checkpoint on a single random seed (hardcoded)
            - produces a confusion matrix displaying the test performance of this model
            - outputs results to csv file: simclr_classification_metrics.csv.

#### simclr_v_baseline_comparer.py:
        Usage:
            python3 simclr_v_baseline_comparer.py
        Functionality:
            -  creates a dot plot with error bars comparing the mean linear evaluation performance of the ImageNet pre-trained ResNet-50 and a custom SimCLR trained ResNet-50 encoder over the ten random seeds.
            - outputs results to csv files: resnet_baseline_linear_eval.csv, simclr_linear_checkpoint_[].csv.

#### yolo_run_analyser.py:
        Usage:
            python3 yolo_run_analyser.py
        Functionality:
        - Given the results of a YOLOv11-n-cls training run, produces confusion matrices displaying the test performance of the model on the dataset
        - outputs results to csv file: yolo_classification_metrics.csv.

#### yolo_simclr_comparer.py
        Usage:
            python3 yolo_simclr_comparer.py
        Functionality:
        - Given the results of a SimCLR trained ResNet-50 evaluation and of a YOLOv11-n-cls training run (both over ten seeds), produces a bar chart comparing the performance of the models across the ten seeds.
        -  outputs results to csv file: comparison_results.csv.

#### evaluate_f4k.py:
        Usage:
            python3 evaluate_f4k.py
        Functionality:
        - Given the results of a SimCLR trained ResNet-50 evaluation (only linear evaluation and partial fine-tuning on a single seed), produces a bar chart visualizing this evaluation. 

#### sign_tester.py:
        Usage:
            python3 sign_tester.py
        Functionality:
        - Given a csv file comparing the test accuracy of SimCLR trained ResNet-50 vs the test accuracy of a supervised YOLOv11-n classifier across a set of matched random seeds, performs a simple statistical sign test.
        - outputs results to csv file: sign_test.csv.