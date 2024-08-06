Semi-Supervised Learning for Apple Detection Using YOLO

Overview:
This project aims to develop and test functions for semi-supervised learning to improve apple detection using the YOLO (You Only Look Once) object detection model. The semi-supervised learning approach leverages both labeled and unlabeled data to enhance the model's performance.

Objectives:
Implement functions to train a YOLO model on labeled data.
Develop methods to generate pseudo-labels for unlabeled data.
Retrain the YOLO model using a mix of labeled and pseudo-labeled data.
Evaluate the effectiveness of the semi-supervised learning approach.

Methodology:
Initial Training on Labeled Data:
Train a YOLO model using a dataset of labeled apple images.
Save the trained model for further use.
Generating Pseudo-Labels:

Use the trained YOLO model to predict labels for a set of unlabeled apple images.
Filter predictions based on confidence scores to ensure high-quality pseudo-labels.
Integrate these pseudo-labels into the existing labeled dataset.
Retraining with Mixed Data:

Combine the original labeled data with the pseudo-labeled data.
Retrain the YOLO model on this mixed dataset to refine its detection capabilities.
Evaluation:

Assess the model's performance using metrics such as precision, recall, and mean average precision (mAP).
Compare the results with the baseline model trained solely on labeled data.
