# PointNet_Superficial_Roughness_Classifier_Hollow_Metal_AM_Parts

This project is based on the <a href="https://github.com/charlesq34/pointnet" target="_blank">PointNet</a> classification model in order to classify superficial quality of metal deposited hollow parts based on the superficial roughness of those parts.

TensorFlow 2.x.x is used adapting some functions from TensorFlow 1 in order to adapt legacy versions of the PointNet classification model to classify 3D models' surfaces.

The pipeline for simulating 3D models, transform and prepare data work as follows:

- Run create_rugosity_from_mesh.py in order to generate superficial rugosity on the base 3D models in 3D_meshes/Original based on pre-defined classes, N BATCH_SIZE 3D models and .csv files for positional and normals data will be created;
- Run segmentation_icp_point_reduction.py in order to create 2048 points segments from the simulated positional data in order to prepare the input of the neural network;
- Run pre_process_data_unified.py to create H5 files that will be used as input to the neural network;
- Finally run train_with_normals.py to train the neural network on the segmented simulated 3D models.
