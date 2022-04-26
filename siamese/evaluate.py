import argparse
import json
import numpy as np
from pathlib import Path

from data import get_zoo_elephants_images_and_labels, get_ELEP_images_and_labels, get_eval_dataset
from train import SiameseModel
from metrics import get_kernel_mask, val, far, pairwise_accuracy

from tensorflow.keras import optimizers

print('start')

# python evaluate.py --weights=/Users/deepakduggirala/Downloads/best_weights --data_dir=/Users/deepakduggirala/Documents/Elephants-dataset-cropped-png-1024

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=1.25,
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

    # cache_file = str(Path(args.data_dir) / 'eval.cache')
    images, labels = get_eval_dataset(get_zoo_elephants_images_and_labels, params,
                                      Path(args.data_dir), cache_file=None, batch_size=32)

    embeddings = siamese_model.predict(images)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=1)

    kernel, mask = get_kernel_mask(labels, embeddings)
    d = float(args.d)
    print('d', d)
    print('Validation rate (VAL): ', val(kernel, mask, d))
    print('False accept rate (FAR): ', far(kernel, mask, d))
    print('pairwise_accuracy: ', pairwise_accuracy(kernel, mask, d))

    for d in np.arange(0.1, 1.4, 0.1):
        print('d', d)
        print('Validation rate (VAL): ', val(kernel, mask, d))
        print('False accept rate (FAR): ', far(kernel, mask, d))
        print('pairwise_accuracy: ', pairwise_accuracy(kernel, mask, d))
        print('\n\n\n')
