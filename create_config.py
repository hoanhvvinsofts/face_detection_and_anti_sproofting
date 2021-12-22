import configparser

config = configparser.ConfigParser()
config['FACEMODEL'] = {'image_size': '112,112',
                        'model': 'insightface/models/model-y1-test2/model,0',
                        'ga_model': '',
                        'threshold': 1.24,
                        'det': 0}

config['EMBEDDINGS_AND_LABELS'] = {'embeddings_path': 'src/outputs/embeddings.pickle',
                                    'labels_path': 'src/outputs/le.pickle'}

config['GPU_ID'] = {'gpu_id': 0}

config["ANTI_SPOOFING_MODEL"] = {'modelv1_path': 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth',
                                 'modelv2_path': 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'}

config["THRESHOLD_PREDICT"] = {'cosine_threshold': 0.4,    
                               'proba_threshold': 0.08,
                               'comparing_num': 5}

config["SAVE_FRAME"] = {'save_frame_number': 10,
                        'temp_folder': 'datasets/unlabel/unknown'}

config["CROP_BOX"] = {'expand_box_ratio': 6,
                      'ratio_min': 0.98,
                      'ratio_max': 1.02}

config["RESOLUTION_FRAME"] = {'frame_width': 1280,    
                                'frame_height': 720}

config["FACE_DETECTION_MODEL"] = {'model_selection': 1,    
                                    'min_detection_confidence': 0.5}

config["BUTTONS"] = {'quit': 'q',
                     'save_frame': 'a'}

config["PROCESSING_FRAME_EVERY"] = {'processing_frame_every': 3}

config["INPUT_SHAPE"] = {'classication': (112, 112),
                        'anti_spoofing': (80, 80)}

config["DATASET"] = {'train_dataset_path': 'datasets/train'}

config["SVM_MODEL"] = {'svm_path': 'src/outputs/model.pkl',
                       'kernel': 'linear',
                       'probability': 'True', 
                       'max_iter': -1}

config["TORCH_DEVICE"] = {'device': 'cuda:0'}

config["DATABASE"] = {'database_name': 'timekeeping',
                      'database_file': 'timekeeping.db'}

with open('config.ini', 'w') as configfile:
    config.write(configfile)