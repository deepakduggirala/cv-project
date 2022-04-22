import argparse
import json
import numpy as np

from data import get_zoo_elephants_images_and_labels, get_ELEP_images_and_labels, get_eval_dataset
from train import SiameseModel
from metrics import get_kernel_mask, val, far, pairwise_accuracy

from tensorflow.keras import optimizers

# python evaluate.py --weights=/Users/deepakduggirala/Downloads/best_weights --data_dir=/Users/deepakduggirala/Documents/Elephants-dataset-cropped-png-1024

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=0.25,
                        help="Squared euclidean distance threshold")
    parser.add_argument('--params', default='hyperparameters/initial_run.json',
                        help="JSON file with parameters")
    parser.add_argument('--data_dir', default='../data/',
                        help="Directory containing the dataset")
    parser.add_argument('--weights', default='best_weights',
                        help="Load the model from the last Checkpoint")
    args = parser.parse_args()

    print(args)

    with open(args.params, 'rb') as f:
        params = json.load(f)

    print(params)

    siamese_model = SiameseModel(params, True)
    # siamese_model.compile(optimizer=optimizers.Adam(params['lr']))
    siamese_model.load_weights(args.weights)

    # input_shape = (None, params['image_size'], params['image_size'], 3)
    # siamese_model.compute_output_shape(input_shape=input_shape)

    images, labels = get_eval_dataset(get_ELEP_images_and_labels, params, args.data_dir)

    embeddings = siamese_model.predict(images)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=1)

    kernel, mask = get_kernel_mask(labels, embeddings)
    print('Validation rate (VAL): ', val(kernel, mask, args.d))
    print('False accept rate (FAR): ', far(kernel, mask, args.d))
    print('pairwise_accuracy: ', pairwise_accuracy(kernel, mask, args.d))
