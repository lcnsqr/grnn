#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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

	// Calcular o erro ou salvar um arquivo com o resultado
	printf("Estimando %d amostras de teste...\n", estim.total);
	if (argc > 1 ){
		// Salvar resultado no arquivo informado
		estimar(&train, &estim, NULL);
		pathSetSave(argv[1], &estim);
		printf("Resultado salvo em %s\n", argv[1]);
	}
	else {
		// Calcular o erro médio para as estimativas
		float errsum = 0;
		estimar(&train, &estim, &errsum);
		printf("Erro médio: %f\n", errsum / (float)estim.total);
	}

	return 0;
}
