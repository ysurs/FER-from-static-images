
landmark_annotation_model = "full"
learning_rate_landmark = 0.01
batch_size_landmark = 32
epochs_landmark = 10
steps_per_epoch_landmark = 30
checkpoint_model_landmark = f'checkpoint_landmarks_model_{landmark_annotation_model}'
filename_reg_lr = f"logs_dict_lr_{lr_model_name}_test1.json"
# preprocess_input_reg_lr = preprocess_input_v1


# Taken from https://github.com/rohan598/Landmark-Aware-Part-based-Ensemble-Transfer-Learning-Network-for-Facial-Expression-Recognition/blob/main/notebooks/summary_notebook.ipynb
# This is the position of landmarks for respective face areas
landmarks_dict = {
    "full":{
        "start_idx":0,
        "end_idx":68,
        "output_size":136
    },
    "mouth":{
        "start_idx":48,
        "end_idx":68,
        "output_size":40
    },
    "eyes":{
        "start_idx":36,
        "end_idx":48,
        "output_size":24
    },
    "nose":{
        "start_idx":27,
        "end_idx":36,
        "output_size":18
    },
    "eyebrows":{
        "start_idx":17,
        "end_idx":27,
        "output_size":20
    },
    "jaw":{
        "start_idx":0,
        "end_idx":17,
        "output_size":34
    }
}



# Contains parameters for landmark annotation model
parameters_landmark_annotation = {
    "output_size":landmarks_dict[landmark_annotation_model]["output_size"],
    "start_idx":landmarks_dict[landmark_annotation_model]["start_idx"],
    "end_idx":landmarks_dict[landmark_annotation_model]["end_idx"],
    "learning_rate_base": learning_rate_landmark,
    "batch_size":batch_size_landmark,
    "epochs":epochs_landmark,
    "steps_per_epoch":steps_per_epoch_landmark,
    "checkpoint":checkpoint_name_reg_lr,
    # "filename":filename_reg_lr,
    "model_name":landmark_annotation_model,
    # "preprocess_input":preprocess_input_reg_lr,
    # "aug_lr":aug_lr_train,
    # "landmarks_predictor":ert_landmarks_predictor
}