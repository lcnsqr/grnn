#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pathio.h"
#include "grnn_gpu.h"

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivos de teste
#define TEST "test.bin"

int main(int argc, char **argv){
	// Identificar dispositivo
	init_gpu();

	struct pathSet train, estim;
	// Carregar arquivo das amostras de treinamento
	pathSetLoad(TRAIN, &train);

	// Arquivo de teste
	pathSetLoad(TEST, &estim);

	printf("Conjunto de treinamento: %d amostras.\n", train.total);
	printf("Dimensões da variável independente: %d\n", train.dim[0]);
	printf("Dimensões da variável dependente:   %d\n", train.dim[1]);

	//float errsum = 0;
	estimar(&train, &estim, NULL);
	//printf("\nErro médio: %f\n\n", errsum / (float)estim.total);

	return 0;
}

