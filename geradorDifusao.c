#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
#include "difusao.h"
#include "pathio.h"

// Constante da equação do calor
#define ALFA 1

// Gerar as soluções por diferenças finitas, após T segundos
#define T 1e-1

// Valores padrão
// Extremidade inicial e final
#define POS0 0
#define POS1 1

// Valores padrão
// Menor e maior valores de leitura
#define MINVAL 0
#define MAXVAL 1

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivo das amostras de teste
#define TEST "test.bin"

// Variável aleatória uniforme em [0,1)
#define RAND ((float)(rand() >> 1)/((RAND_MAX >> 1) + 1))

// Função de preenchimento da condição inicial.
void inicial(Matrix *w){
	// Pontos na curva inicial
	int dim = w->rows - 1;
	// Gerar uma curva aleatória
	w->_[0] = RAND * MAXVAL;
	for (int i = 1; i < dim; i++){
		w->_[i] = w->_[i-1] - .25 + .5 * RAND;
		if ( w->_[i] < MINVAL ) w->_[i] = MINVAL;
		if ( w->_[i] > MAXVAL ) w->_[i] = MAXVAL;
	}
	// Valor da derivada em relação a x no extremo 
	// b definido pelo último valor do vetor (coluna)
	w->_[dim] = 0;
}

int main (int argc, char **argv){
	// Opções da linha de comando
	int total = 20480000;
	int totaltest = 1024;
	int dim = 6;
	for(int i = 1; i < argc; i++){
		switch (argv[i][1]){
		case 't':
			// Total de amostras
			total = atoi(argv[i+1]);
		break;
		case 'e':
			// Total de amostras de teste
			totaltest = atoi(argv[i+1]);
		break;
		case 'd':
			// Dimensões da amostra
			dim = atoi(argv[i+1]);
		break;
		}
	}

	// Semente aleatória
	srand((unsigned int)time(NULL));
	// Utilizar uma semente fixa para gerar sempre os mesmos conjuntos
	srand(0);

	// Conjunto de treinamento
	struct pathSet train;
	// Usar float (4 bytes)
	train.type = 0x0f;
	train.total = total;
	train.vertices = 2;
	train.dim = (unsigned int*)malloc(2*sizeof(unsigned int));
	train.dim[0] = dim;
	train.dim[1] = dim;
	// Tamanho de um caminho (soma das dimensões dos vértices)
	train.size = train.total*(train.dim[0]+train.dim[1])*sizeof(float);
	train.data.f = (float*)malloc(train.size);
	
	// Vetores antes e depois (condição inicial e vetor resultante)
	Matrix w[2];
	// Alocar e incluir dimensão para a condição de Neuman
	mtrxBuild(&w[0], dim + 1, 1);
	mtrxBuild(&w[1], dim + 1, 1);
	// Gerar cada amostra e copiar para o conjunto de treinamento
	printf("Gerando %d amostras de treinamento...\n", train.total);
	for (int i = 0; i < train.total; i++){
		// Preencher condições iniciais
		inicial(&w[0]);
		// Gerar as soluções por diferenças finitas, após T segundos
		gerar(w, ALFA, POS0, POS1, T);
		// Copiar v.i. para posição correspondente na memória do conjunto
		for (int j = 0; j < train.dim[0]; j++){
			train.data.f[i + j * train.total] = w[0]._[j];
		}
		// Variável dependente
		for (int j = 0; j < train.dim[1]; j++){
			train.data.f[train.total*train.dim[0] + i + j * train.total] = w[1]._[j];
		}
	}
	// Salvar conjunto de treinamento
	pathSetSave(TRAIN, &train);
	free(train.dim);
	free(train.data.f);

	// Conjunto de teste
	struct pathSet test;
	// Usar float (4 bytes)
	test.type = 0x0f;
	test.total = totaltest;
	test.vertices = 2;
	test.dim = (unsigned int*)malloc(2*sizeof(unsigned int));
	test.dim[0] = dim;
	test.dim[1] = dim;
	test.size = test.total*(train.dim[0]+train.dim[1])*sizeof(float);
	test.data.f = (float*)malloc(test.size);
	
	// Gerar cada amostra e copiar para o conjunto de teste
	printf("Gerando %d amostras de teste...\n", test.total);
	for (int i = 0; i < test.total; i++){
		// Preencher condições iniciais
		inicial(&w[0]);
		// Gerar as soluções por diferenças finitas, após T segundos
		gerar(w, ALFA, POS0, POS1, T);
		// Copiar v.i. para posição correspondente na memória do conjunto
		for (int j = 0; j < test.dim[0]; j++){
			test.data.f[i + j * test.total] = w[0]._[j];
		}
		// Variável dependente
		for (int j = 0; j < test.dim[1]; j++){
			test.data.f[test.total*test.dim[0] + i + j * test.total] = w[1]._[j];
		}
	}

	// Salvar conjunto de treinamento
	pathSetSave(TEST, &test);
	free(test.dim);
	free(test.data.f);

	return 0;
}
