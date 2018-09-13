#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

STEPS = 5000

# Cada linha do arquivo CSV contém as componentes da variável independente e dependente.
# Nomes das componentes e os tipos de seus respectivos valores (float)
csv_header = collections.OrderedDict([
    ("i0", [0.0]),
    ("i1", [0.0]),
    ("i2", [0.0]),
    ("i3", [0.0]),
    ("d0", [0.0]),
    ("d1", [0.0]),
    ("d2", [0.0]),
    ("d3", [0.0])
])
types = collections.OrderedDict((key, type(value[0])) for key, value in csv_header.items())

def dataset():
  """Carrega os conjuntos amostrais de treinamento e teste.

  O conjunto amostral produz um par (features_dict, label).

  Retorna:
    Um par (train,test) de conjuntos amostrais
  """
  # Conversão das linhas do arquivo CSV
  def decode_line(line):
    """Converte uma linha csv num par (features_dict,label)."""
    # Decodificar a linha para um tuple baseado no tipos em csv_header.values().
    items = tf.decode_csv(line, list(csv_header.values()))

    # Converte as chaves e itens para um dict.
    pairs = zip(csv_header.keys(), items)

    # Features
    features_dict = dict(pairs)

    # Label
    label_list = []
    label_list.append(features_dict.pop("d0"))
    label_list.append(features_dict.pop("d1"))
    label_list.append(features_dict.pop("d2"))
    label_list.append(features_dict.pop("d3"))
    label = tf.convert_to_tensor(label_list)

    return features_dict, label

  train = (tf.data.TextLineDataset("train.csv").map(decode_line))
  test = (tf.data.TextLineDataset("test.csv").map(decode_line))

  return train, test

def main(argv):
  """Construir, treinar e avaliar o modelo."""
  assert len(argv) == 1
  (train, test) = dataset()

  # Construir a função de entrada do treinamento
  def input_train():
    return (train.shuffle(100000).batch(128).repeat().make_one_shot_iterator().get_next())

  # Construir a função de entrada da validação
  def input_test():
    return (test.shuffle(1000).batch(128).make_one_shot_iterator().get_next())

  feature_columns = [
      tf.feature_column.numeric_column(key="i0"),
      tf.feature_column.numeric_column(key="i1"),
      tf.feature_column.numeric_column(key="i2"),
      tf.feature_column.numeric_column(key="i3"),
  ]

  # Estimador DNNRegressor, com duas camadas internas de 2x4 unidades
  model = tf.estimator.DNNRegressor(hidden_units=[4,4], feature_columns=feature_columns, label_dimension=4, model_dir="model")

  # Treinar o modelo
  model.train(input_fn=input_train, steps=STEPS)

  # Avaliar o conjunto de teste
  eval_result = model.evaluate(input_fn=input_test)

  # A chave "average_loss" armazena o Mean Squared Error (MSE)
  average_loss = eval_result["average_loss"]

  # Exibir a raiz (RMSE).
  print()
  print("Erro para o conjunto de teste (RMSE): {:.6f}".format(average_loss**0.5))
  print()


if __name__ == "__main__":
  # Exibir o nível de log "INFO" gerado pelo estimador
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
