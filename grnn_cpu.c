#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "pathio.h"

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivos de teste
#define TEST "test.bin"

// Estimar a variável dependente
// train: Conjunto de treinamento
// total: total de pares no treinamento
// dim: Dimensões da variável independente e dependente
// x: Variável independente lida
// y: Estimativa da variável dependente
// s: Parâmetro da regressão
void estimativa(float *train, unsigned int total, unsigned int *dim, float *x, float *y, float s){
	// Acumuladores do numerador e denominador do estimador
	// para cada dimensão da variável dependente
	float *numer = (float*)malloc(dim[1]*sizeof(float));
	float *denom = (float*)malloc(dim[1]*sizeof(float));
	for (int c = 0; c < dim[1]; c++){
		numer[c] = 0;
		denom[c] = 0;
	}
	// Fator comum para cada amostra
	float f;
	// Quadrado da distância euclidiana entre a amostra e o estimando
	float d;
	// Iterar em cada amostra de treinamento
	for (int i = 0; i < total; i++){
		// Computar o fator comum da i-esima amostra
		d = 0;
		for (int j = 0; j < dim[0]; j++){
			d += pow(train[i + j * total] - x[j], 2);
		}
		f = exp( - d / s );
		// Iterar para cada componente de y
		for (int c = 0; c < dim[1]; c++){
			// Numerador da fração para o c-ésimo componente
			numer[c] += train[total * dim[0] + c * total + i] * f;
			// Denominador da fração
			denom[c] += f;
		}
	}
	// Estimativa com verificação de divisão por zero
	// para cada dimensão da variável dependente
	for (int c = 0; c < dim[1]; c++){
		if ( denom[c] != 0 ){
			y[c] = numer[c] / denom[c];
		}
		else {
			// Falha na operação
			puts("Divisão por zero\n");
			exit(EXIT_FAILURE);
		}
	}
	free(numer);
	free(denom);
}

void estimar(struct pathSet *train, struct pathSet *estim, float *errsum){
	// Vetor da variável independente 
	float *x = (float*)malloc(sizeof(float)*train->dim[0]);
	// Vetor da estimativa 
	float *y = (float*)malloc(sizeof(float)*train->dim[1]);
	// Erro da estimativa
	float err = 0;
	// Tamanho das dimensões das variáveis
	unsigned int *dim = train->dim;
	// Parâmetro sigma (variância)
	float sigma = 1.0/log(train->total);
	// Expressão envolvendo sigma no numerador do fator comum é constante
	float s = 2*pow(sigma,2);
	// Iterar em todo o conjunto de teste
	for (int i = 0; i < estim->total; i++){
		for (int j = 0; j < dim[0]; j++){
			x[j] = estim->data.f[i + estim->total * j];
		}
		estimativa(train->data.f, train->total, train->dim, x, y, s);
		// Sem endereço para o erro somado, substituir valores no conjunto estimado
		if ( errsum == NULL ){
			// Escrever estimativa
			for (int j = 0; j < dim[1]; j++){
				estim->data.f[estim->total*dim[0] + i + estim->total*j] = y[j];
			}
		}
		else {
			// Erro da estimativa
			err = 0;
			for (int j = 0; j < dim[1]; j++){
				err += pow(estim->data.f[estim->total*dim[0] + i + estim->total*j] - y[j], 2);
			}
			err = sqrt(err);
			// Erro acumulado
			*errsum += err;
		}
	}
	free(x);
	free(y);
}

int main(int argc, char **argv){
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
