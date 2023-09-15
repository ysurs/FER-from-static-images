# FER-from-static-images
Implementation of a paper: Landmark-Aware and Part-based Ensemble Transfer Learning Network for Facial Expression Recognition from Static images in pytorch



## At present, the following files contain this code:

1. baseline_model.py : Contains code for the baseline model which is used to compare with the proposed model. This has to be tested.

2. callback.py : Contains code for the custom callback function with early stopping criteria. Needs to be tested and verified if this is the right thing to do.

3. check.ipynb : Contains code for experimentions, checking classes and methods. 

4. components : Contains feature extractor code, landmark localisation and expression classification.

5. config.py : Contains hyperparameter configuration.

6. dataset_for_landmark : Going to contain custom dataset class for landmark localisation model.

7. dataset.py : Contains custom dataset class for reading the dataset and getting it ready for preprocessing.

8. landmark_train.py : Contains custom landmark training code. To be completed and tested.

9. Planning.txt : Contains observations, questions regarding the implementation.

10. preprocessing.py : Contains preprocessing code including clahe, single shot detection, landmark annotation.

**Note:** This is under development so file naming and organization can change.