
# COSCO:A Sharpness Aware Training Framework for Few-Shot Multivariate Time Series Classification
## Abstract
Time series classification is an important task with widespread applications in many different domains. Recently, deep neural networks have achieved state-of-the-art performance in time series classification, but often require large datasets and expert-labeled supervision for effective model training. In data scarcity situations, deep learning models often experience a significant decline. In this paper, we propose a training framework named COSCO to mitigate the decline in performance from a loss function and optimization perspective for the few-shot multivariate time series classification problem, where only a few samples are available. Particularly, we propose to apply a prototypical loss function and the sharpness-aware minimization (SAM) technique to enhance the generalization ability for deep learning classifier in few-shot multivariate time series classification problem. Such optimization techniques can be arbitrarily used in any deep learning models. We demonstrate the effectiveness of our method with ResNet backbone on few-shot benchmark datasets formed by UCR classification data.


## Environment 

To run this project offer the option of using a Jupiter Notebook ```COSCO.ipynb``` for plug and play . 

You are also able to run locally by cloning the repository and installing the dependencies.

```bash
  git clone https://github.com/JRB9/COSCO.git
  cd COSCO
  pip install -r requirements.txt
```



## Datasets

For plug and play we have provided the Full Datasets, 1-shot and 10-shot versions in this repository in the Dataset folder. These datsets were gathered from [UCR Time Series Classification Archive](https://timeseriesclassification.com/dataset.php)
. The list of multivariate datasets are as follows:
```bash
ArticularyWordRecognition, BasicMotions, CharacterTrajectories, EigenWorms, Epilepsy, EthanolConcentration, FaceDetection, FingerMovements, HandMovementDirection, Heartbeat, JapaneseVowels, Libras, MotorImagery, NATOPS, PEMS-SF, PenDigits, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, UWaveGestureLibrary. 
```
## Reproducing
### Using Jupiter Notebook
Run ```COSCO.ipynb``` in any Jupiter Notebook environment.
### Running Locally
Execute```run.py``` with the following configurable arguments:
```bash
python run.py --dataset <dataset> --model <model> --lr <learning_rate> --rho <rho> --nEpoch <number_of_epochs> --shot <shots> --normalize <True/False> --sam <True/False> --optimizer <sgd/adam> --prototypical_loss <True/False> --prototypical_loss_type <neg/sim/cos/negexp> --save_dir <save_directory> --save_name <save_filename>
```
#### Argument Descriptions

- **lr**: Learning rate for the optimizer (e.g., `0.001`).
- **rho**: Momentum parameter for optimization (e.g., `0.9`).
- **nEpoch**: Number of epochs for training (e.g., `100`).
- **dataset**: The name of the dataset to use (e.g., "BasicMotions").
- **shot**: Number of shots (support examples) per class (`1` or `10`).
- **normalize**: Whether to normalize the input data (`True` or `False`).
- **model**: The type of baseline model to use (`"resnet"` or `"tapnet"`).
- **sam**: Whether to use Sharpness-Aware Minimization (SAM) (`True` or `False`).
- **optimizer**: The optimizer type for training (`"sgd"` or `"adam"`).
- **prototypical_loss**: Whether to apply prototypical loss during training (`True` or `False`).
- **prototypical_loss_type**: The type of prototypical loss to use (`"neg"`, `"sim"`, `"cos"`, `"negexp"`).
- **save_dir**: Directory path to save the output (e.g., `"/content/classification_data/"`).
- **save_name**: File name for saving results (e.g., `"results.csv"`).

##### Example Usage

```bash
python run.py --dataset BasicMotions --model resnet --lr 0.001 --rho 0.9 --nEpoch 100 --shot 1 --normalize False --sam True --optimizer adam --prototypical_loss True --prototypical_loss_type neg --save_dir /content/classification_data/ --save_name results.csv
```




## Appendix

Submitted & accepted for publication in the CIKM '24 conference.

If you use this code or our methods in your research, please cite our paper:

Jesus Barreda, Ashley Gomez, Ruben Puga, Kaixiong Zhou, and Li Zhang. 2024. COSCO: A Sharpness-Aware Training Framework for Few-shot Multivariate Time Series Classification. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24). Association for Computing Machinery, New York, NY, USA, 3622–3626. https://doi.org/10.1145/3627673.3679891


