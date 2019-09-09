import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from metrics import Recall
from embedding import Embedding
from preprocessor import Preprocessor


def main(test_path, result_path):
    # -----------------------------------load config
    with open('config.json') as f:
        config = json.load(f)

    # -----------------------------------load embedding
    logging.info('loading embedding...')
    with open('attn_embedding.pkl', 'rb') as f:
        embedding = pickle.load(f)
        config['model_parameters']['embedding'] = embedding.vectors
        
    # -----------------------------------update embedding used by preprocessor
    preprocessor = Preprocessor(None)
    preprocessor.embedding = embedding

    # -----------------------------------get dataset
    logging.info('Processing test from test.pkl')
    test = preprocessor.get_dataset(test_path, 6, {'n_positive': -1, 'n_negative': -1, 'shuffle': False})
    test.shuffle = False

    # -----------------------------------make model
    from example_predictor import ExamplePredictor
    PredictorClass = ExamplePredictor
    predictor = PredictorClass(metrics=[], **config['model_parameters'])

    # -----------------------------------load model
    logging.info('loading model from {}'.format('model.pkl.4'))
    predictor.load('attn_model.pkl.4')

    # -----------------------------------predicting
    logging.info('predicting...')
    predicts = predictor.predict_dataset(test, test.collate_fn)

    # -----------------------------------save csv
    write_predict_csv(predicts, test, result_path)


def write_predict_csv(predicts, data, output_path, n=10):
    outputs = []
    for predict, sample in zip(predicts, data):
        candidate_ranking = [
            {
                'candidate-id': oid,
                'confidence': score.item()
            }
            for score, oid in zip(predict, sample['option_ids'])
        ]

        candidate_ranking = sorted(candidate_ranking,
                                   key=lambda x: -x['confidence'])
        best_ids = [candidate_ranking[i]['candidate-id']
                    for i in range(n)]
        outputs.append(
            ''.join(
                ['1-' if oid in best_ids
                 else '0-'
                 for oid in sample['option_ids']])
        )

    logging.info('Writing output to {}'.format(output_path))
    with open(output_path, 'w') as f:
        f.write('Id,Predict\n')
        for output, sample in zip(outputs, data):
            f.write(
                '{},{}\n'.format(
                    sample['id'],
                    output
                )
            )


if __name__ == '__main__':
    test_path = sys.argv[1]
    result_path = sys.argv[2]
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(test_path, result_path)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
