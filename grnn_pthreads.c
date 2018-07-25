#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pathio.h"

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivos de teste
#define TEST "test.bin"

// Quantidade de threads
#define NUM_THREADS	8

// Estrutura para o contexto de cada thread que 
// computa uma parcial do conjunto de treinamento
struct TrainPart {
	// Endereço da parte donjunto de treinamento
	float *train;
	// Endereço da parte donjunto de treinamento
	// com as variáveis dependentes
	float *train_y;
	// Total de pares no treinamento
	unsigned int total;
	// Total de pares no treinamento parcial
	unsigned int totalPart;
	// Dimensões da variável independente e dependente
	unsigned int *dim;
	// Variável independente lida
	float *x;
	// Parâmetro da regressão
	float s;
	// Acumuladores do numerador e denominador
	float *numer;
	float *denom;
};

// Thread para computar parcial do conjunto de treinamento
void *estimPart(void *voidTrainPart){
	// Estrutura da parcial do treinamento
	struct TrainPart *tp = (struct TrainPart*)voidTrainPart;
	// Fator comum para cada amostra
	float f;
	// Quadrado da distância euclidiana entre a amostra e o estimando
	float d;
	// Iterar em cada amostra de treinamento para o conjunto parcial
	for (int i = 0; i < tp->totalPart; i++){
		// Computar o fator comum da i-esima amostra
		d = 0;
		for (int j = 0; j < tp->dim[0]; j++){
			d += pow(tp->train[i + j * tp->total] - tp->x[j], 2);
		}
		f = exp( - d / tp->s );
		// Iterar para cada componente de y
		for (int c = 0; c < tp->dim[1]; c++){
			// Numerador da fração para o c-ésimo componente
			tp->numer[c] += tp->train_y[c * tp->total + i] * f;
			// Denominador da fração
			tp->denom[c] += f;
		}
	}
	pthread_exit((void*)voidTrainPart);
}

// Estimar a variável dependente
// train: Conjunto de treinamento
// total: total de pares no treinamento
// dim: Dimensões da variável independente e dependente
// x: Variável independente lida
// y: Estimativa da variável dependente
// s: Parâmetro da regressão
void estimativa(float *train, unsigned int total, unsigned int *dim, float *x, float *y, float s){
	// Contexto para cada thread
	struct TrainPart tp[NUM_THREADS];
	// Acumuladores do numerador e denominador 
	// do estimador para cada thread
	float *aNumer[NUM_THREADS];
	float *aDenom[NUM_THREADS];
	// Configurar contexto para cada thread
	for (int t = 0; t < NUM_THREADS; t++){
		// Acumuladores para cada dimensão da variável dependente
		aNumer[t] = (float*)malloc(dim[1]*sizeof(float));
		aDenom[t] = (float*)malloc(dim[1]*sizeof(float));
		for (int c = 0; c < dim[1]; c++){
			aNumer[t][c] = 0;
			aDenom[t][c] = 0;
		}
		tp[t].train = &train[t*(total/NUM_THREADS)];
		tp[t].train_y = &train[total*dim[0] + t*(total/NUM_THREADS)];
		tp[t].total = total;
		tp[t].totalPart = total / NUM_THREADS;
		tp[t].dim = dim;
		tp[t].x = x;
		tp[t].s = s;
		tp[t].numer = aNumer[t];
		tp[t].denom = aDenom[t];
	}

	// Threads para cada subconjunto de treinamento
   pthread_t thread[NUM_THREADS];
   pthread_attr_t attr;
   int pthread_return;
   void *pthread_status;

	// Iniciar threads
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(int t = 0; t < NUM_THREADS; t++){
		pthread_return = pthread_create(&thread[t], &attr, estimPart, (void *)&tp[t]); 
		if (pthread_return) {
			printf("ERROR; return code from pthread_create() is %d\n", pthread_return);
			exit(EXIT_FAILURE);
		}
	}

	// Liberar atributo e aguardar threads
	pthread_attr_destroy(&attr);
	for(int t = 0; t < NUM_THREADS; t++){
		pthread_return = pthread_join(thread[t], &pthread_status);
		if (pthread_return){
			printf("ERROR; return code from pthread_join() is %d\n", pthread_return);
			exit(EXIT_FAILURE);
		}
		//printf("Finalizado thread %d\n",t);
	}

	// Acumuladores do numerador e denominador 
	// para a variável dependente estimada
	float *numer = (float*)malloc(dim[1]*sizeof(float));
	float *denom = (float*)malloc(dim[1]*sizeof(float));
	for (int c = 0; c < dim[1]; c++){
		numer[c] = 0;
		denom[c] = 0;
	}
	for (int t = 0; t < NUM_THREADS; t++){
		// Estimativa com verificação de divisão por zero
		// para cada dimensão da variável dependente
		for (int c = 0; c < dim[1]; c++){
			numer[c] += aNumer[t][c];
			denom[c] += aDenom[t][c];
		}
		free(aNumer[t]);
		free(aDenom[t]);
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
		// Construir variável independente envolvida 
		// a partir dos dados dispostos na memória
		for (int j = 0; j < dim[0]; j++){
			x[j] = estim->data.f[i + estim->total * j];
		}
		// Gerar estimativa
		estimativa(train->data.f, train->total, dim, x, y, s);
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
