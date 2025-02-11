# Self-Supervised Out-of-Distribution Detection - Metal Implants and Other Anomaly

### Setting up your environment
1. Create an environment with .yml file

    ```conda create -n env python=3.12```
    ```conda activate env```
    ```pip install -r requirements.txt```

2. Make changes to your enviroment. Copy and paste ```mae.py``` and ```mae_recon.py``` in your environment (```~/miniconda3/envs/mae/lib/python3.12/site-packages/vit_pytorch/```)


### Inference
1. The inference is done in two steps
    - Inference Generation : Here the difference map and the reconstructions of the slice/series is generated and saved as numpy arrays in the specified location
    - Inference Evaluation : Here the saved difference map the slice/series is used to make the prediction if it's an anomaly or not
#### Slice-level Inference
1. To run slice-level inference, you need a ```.csv``` file with a list of all the paths to your slices you want to run inference on under the column ```Path```

2. First, the difference map and the reconstructions of the slices are generated and saved in the specified folder

    ```python inference/slice-level-inference/InferenceGenerate.py --config_file inference/slice-level-inference/config.ini```

3. Run the inference evaluation which outputs a csv file with the predictions

    ```python inference/slice-level-inference/InferenceEvaluate.py --config_file inference/slice-level-inference/config.ini```

#### Series-level Inference
1. The series-level inference requires CT series as resampled NIFTIs. For this inference you need ```.csv``` file with a list of all paths to NIFTI volumes you want to run inference on under the column ```NIFTIPath```

2. First, resampled the NIFTIs

    ```python NIFTIResampled.py --config_file inference/series-level-inference/config.ini``

3. Now run the inference generation code. This would required the output csv from the resampling step

    ```python series-level-inference/InferenceGenerate.py --config_file inference/series-level-inference/config.ini```

4. Now run the inferece evaluation

    ```python series-level-inference/InferenceEvaluate.py --config_file inference/series-level-inference/config.ini```

### Training
Both slice level model and series level model are trained in a similar fashion slice by slice. The only difference is the input image size and the patch size. One can choose ```series_config.ini``` for training a series level model or ```slice_config.ini``` for training a slice level model

```python main.py --config_file <name of config file>```

