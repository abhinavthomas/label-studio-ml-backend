import json
import logging
import os
import random

import spacy
from spacy.training import Example
from tqdm.auto import tqdm

from label_studio_ml.model import LabelStudioMLBase

logging.basicConfig(level=logging.INFO)


class SimpleNER(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(SimpleNER, self).__init__(**kwargs)

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Labels> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Labels'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        if not self.train_output:
            # If there is no trainings, define cold-started the simple spaCy NER model
            self.reset_model()
            # This is an array of <Labels> labels
            self.labels = self.info['labels']
            # Initialized the ner model with labels
            list(map(self.ner.add_label, self.labels))
            print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(
                    self.labels)
            ))
        else:
            # otherwise load the model from the latest training results
            self.model_file = self.train_output['model_file']
            self.model = spacy.load(self.model_file)

            # and use the labels from training outputs
            self.labels = self.train_output['labels']
            print('Loaded from train output with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(
                    self.labels)
            ))

    def reset_model(self):
        self.model = spacy.blank("en")
        self.model.add_pipe("ner")
        self.ner = self.model.get_pipe("ner")
        self.new_model = True

    def predict(self, tasks, **kwargs):
        # collect input texts
        predictions = []
        for task in tasks:
            doc = self.model(task['data'][self.value])
            # get named entities
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'labels',
                'value': {"start": ent.start_char, "end": ent.end_char, "text": ent.text, 'labels': [ent.label_]}
            } for ent in doc.ents]
            predictions.append({'result': result})
        print(predictions)
        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        train_data = []
        _labels = []
        # train the model
        self.reset_model()
        if self.new_model:
            optimizer = self.model.begin_training()
        else:
            optimizer = self.model.resume_training()

        for completion in completions:
            # get input text from task data
            if completion['annotations'][0].get('skipped') or completion['annotations'][0].get('was_cancelled'):
                continue
            # get an annotation
            output_labels = []
            for annotation in completion['annotations']:
                for result in annotation['result']:
                    start = result['value']['start']
                    end = result['value']['end']
                    for label in result['value']['labels']:
                        output_labels.append((start, end, label))
                        _labels.append(label)

            train_data.append((completion['data'][self.value], {
                              'entities': output_labels}))
        new_labels = set(_labels)
        if len(new_labels) != len(self.labels):
            self.labels = list(sorted(new_labels))
            print('Label set has been changed:' + str(self.labels))
        # Training for 30 iterations
        for _ in tqdm(range(30)):
            random.shuffle(train_data)
            for raw_text, entities in train_data:
                doc = self.model.make_doc(raw_text)
                example = Example.from_dict(doc, entities)
                self.model.update([example], sgd=optimizer)

        # save spaCy pipeline to model file
        model_file = os.path.join(workdir, 'model')
        self.model.to_disk(model_file)
        train_output = {
            'labels': self.labels,
            'model_file': model_file
        }
        return train_output
