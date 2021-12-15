import os
import pickle

import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer


class SimpleTextClassifier(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(SimpleTextClassifier, self).__init__(**kwargs)

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Taxonomy'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        if not self.train_output:
            # If there is no trainings, define cold-started the simple TF-IDF text classifier
            self.reset_model()
            # This is an array of <Choice> labels
            self.labels = self.info['labels']
            # make some dummy initialization
            self.model.fit(X=self.labels, y=[[i]
                           for i in range(len(self.labels))])
            print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(
                    self.labels)
            ))
        else:
            # otherwise load the model from the latest training results
            self.model_file = self.train_output['model_file']
            with open(self.model_file, mode='rb') as f:
                self.model = pickle.load(f)
            # otherwise load the MLB from the latest training results
            mlb_file = self.train_output['mlb_file']
            with open(mlb_file, mode='rb') as f:
                self.mlb = pickle.load(f)
            # and use the labels from training outputs
            self.labels = self.train_output['labels']
            print('Loaded from train output with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(
                    self.labels)
            ))

    def reset_model(self):
        self.model = make_pipeline(TfidfVectorizer(
            ngram_range=(1, 3)), MultiOutputClassifier(RandomForestClassifier(random_state=1), n_jobs=-1))
        self.mlb = MultiLabelBinarizer()

    def predict(self, tasks, **kwargs):
        # collect input texts
        input_texts = []
        for task in tasks:
            input_texts.append(task['data'][self.value])

        # get model predictions
        one_hot_encoded = self.model.predict(input_texts)
        predicted_label_indices = self.mlb.inverse_transform(one_hot_encoded)
        predictions = []
        for idx in predicted_label_indices:
            predicted_label = [[self.labels[i]] for i in idx]
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'taxonomy',
                'value': {'taxonomy': predicted_label}
            }]
            predictions.append({'result': result})

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}

        for completion in completions:
            # get input text from task data
            print(completion)
            if completion['annotations'][0].get('skipped') or completion['annotations'][0].get('was_cancelled'):
                continue

            input_text = completion['data'][self.value]
            input_texts.append(input_text)

            # get an annotation
            output_label = list(map(
                lambda x: x[0], completion['annotations'][0]['result'][0]['value']['taxonomy']))
            output_labels.append(output_label)
            output_label_idxs = [label2idx[_label] for _label in output_label]
            output_labels_idx.append(output_label_idxs)

        new_labels = set(
            [_label for _labels in output_labels for _label in _labels])
        if len(new_labels) != len(self.labels):
            self.labels = list(sorted(new_labels))
            print('Label set has been changed:' + str(self.labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [[label2idx[label]
                                  for label in labels] for labels in output_labels]

        # train the model
        self.reset_model()
        print("New label binarizer has been fitted")
        mlb_outputs = self.mlb.fit_transform(output_labels_idx)
        self.model.fit(input_texts, mlb_outputs)

        # save output resources
        model_file = os.path.join(workdir, 'model.pkl')
        with open(model_file, mode='wb') as fout:
            pickle.dump(self.model, fout)
        mlb_file = os.path.join(workdir, 'mlb.pkl')
        with open(mlb_file, mode='wb') as fout:
            pickle.dump(self.mlb, fout)

        train_output = {
            'labels': self.labels,
            'model_file': model_file,
            'mlb_file': mlb_file
        }
        return train_output
