# Self-Supervised Out-of-Distribution Detection - Metal Implants and Other Anomaly

### Setting up your environment
1. Create an environment with .yml file

    ``` conda env create -f environment.yml```

2. Make changes to your enviroment. Copy and paste ```mae.py``` and ```mae_recon.py``` in your environment (```~/miniconda3/envs/mae/lib/python3.12/site-packages/vit_pytorch/```)

3. The model weights will be updated here

### Inference
1. The inference is done in two steps
    - Inference Generation : Here the difference map and the reconstructions of the slice/series is generated and saved as numpy arrays in the specified location
    - Inference Evaluation : Here the saved difference map the slice/series is used to make the prediction if it's an anomaly or not
#### Slice-level Inference
1. To run slice-level inference, you need a ```.csv``` file with a list of all the paths to your slices you want to run inference on under the column ```Path```

2. First, the difference map and the reconstructions of the slices are generated and saved in the specified folder

    ```python slice-level-inference/InferenceGenerate.py --csv_file <path to csv file with 'Path' column> --diff_write_pth <path to write the difference map> --recon_write_path <path to write the reconstructions> --device <gpu device number> --model_weights <path to model weights>```

3. Run the inference evaluation which outputs a csv file with the predictions

    ```python slice-level-inference/InferenceEvaluate.py ---diff_write_path <path where the difference maps are> --csv_file <path to csv file with 'Path' column> --write_path <path to write the prediction as csv>```

#### Series-level Inference
1. The series-level inference requires CT series as resampled NIFTIs. For this inference you need ```.csv``` file with a list of all paths to NIFTI volumes you want to run inference on under the column ```NIFTIPath```

2. First, resampled the NIFTIs

    ```python NIFTIResampled.py ---input_csv_path <path to input csv file with NIFTIPath column>---output_csv_path <path to write the output csv file> ---nifti_write_path <path to write the resampled NIFTIs>```

3. Now run the inference generation code. This would required the output csv from the resampling step

    ```python series-level-inference/InferenceGenerate.py --csv_file <path to output csv from the resampling> --diff_write_pth <path to write the difference map> --recon_write_path <path to write the reconstructions> --device <gpu device number> --model_weights <path to model weights>```

4. Now run the inferece evaluation

    ```python series-level-inference/InferenceEvaluate.py ---diff_write_path <path where the difference maps are> --csv_file <path to csv file with 'Path' column> --write_path <path to write the prediction as csv>```

