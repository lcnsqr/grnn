#!/usr/bin/env python3

import collections
import numpy as np
import tensorflow as tf
import os

def main(argv):
  """Construir, treinar e avaliar o modelo."""
  
  # Os argumentos do comando são os parâmetros de treinamento
  batch = int(argv[1])
  steps = int(argv[2])
  epoch = int(argv[3])

  # Cada linha do arquivo CSV contém as componentes da variável independente e dependente
  totalTrain = 0
  totalTest = 0
  # As colunas são as mesmas para o conjunto de treinamento e de teste
  cols = []
  with open("train.csv") as csv:
      for h in csv.readline().rstrip().split(","):
          cols.append(h)
      totalTrain = sum(1 for line in csv)
  with open("test.csv") as csv:
      totalTest = sum(1 for line in csv) - 1

  # Colunas das componentes da variável independente
  featureCols = []
  # Colunas das componentes da variável dependente
  labelCols = []
  for h in cols:
      if 'x' == h[0]:
          featureCols.append(h)
      if 'y' == h[0]:
          labelCols.append(h)


  def dataset():
    """
    O elemento do conjunto amostral é um par (features_dict, label)

    Retorna:
      Um par (train,test) de conjuntos amostrais
    """
    # Conversão das linhas do arquivo CSV
    def decode_line(line):
      # Converter a linha usando uma lista de valores padrão
      pair = tf.decode_csv(line, len(cols)*[[]])
      return dict(zip(featureCols, pair[:len(featureCols)])), tf.convert_to_tensor(pair[-len(labelCols):])

    train = tf.data.TextLineDataset("train.csv").skip(1).map(decode_line)
    test = tf.data.TextLineDataset("test.csv").skip(1).map(decode_line)

    return train, test

  (train, test) = dataset()

  # Construir a função de entrada do treinamento
  def input_train():
    #return train.shuffle(totalTrain).batch(batch).make_one_shot_iterator().get_next()
    return train.batch(batch).make_one_shot_iterator().get_next()

  # Construir a função de entrada da validação
  def input_test():
    return test.batch(batch).make_one_shot_iterator().get_next()

  # Estimador DNNRegressor
  model = tf.estimator.DNNRegressor(
    # Camadas intermediárias
    hidden_units=[4], 
    # Colunas da variável independente (features)
    feature_columns=list(map(lambda h: tf.feature_column.numeric_column(key=h), featureCols)), 
    # Dimensões da variável dependente (label)
    label_dimension=len(labelCols), 
    # Diretório para salvar o modelo
    model_dir="model"
  )

  # Treinar o modelo indicando o número de passagens
  model.train(input_fn=input_train, steps=steps)

  # Avaliar o conjunto de teste
  eval_result = model.evaluate(input_fn=input_test)

  # A chave "average_loss" armazena o Mean Squared Error (MSE)
  average_loss = eval_result["average_loss"]

  # Exibir a raiz (RMSE).
  #print("Total: {:d}, Batch: {:d}, steps: {:d}".format(totalTrain, batch, steps))
  #print("RMSE em {:d} amostras de teste: {:.6f}".format(totalTest, average_loss**0.5))
  #print()
  print("{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.6f}".format(totalTrain, totalTest, batch, steps, epoch, average_loss**0.5))

if __name__ == "__main__":
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  tf.app.run(main=main)
