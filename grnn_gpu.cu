#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pathio.h"
#include "grnn_gpu.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

// Arquivo das amostras
#define TRAIN "train.bin"
// Arquivos de teste
#define TEST "test.bin"

int main(int argc, char **argv){
	// Opções da linha de comando
	const char* outfile = NULL;
	float ss = 1;
	for(int i = 1; i < argc; i++){
		switch (argv[i][1]){
		case 'o':
			// Salvar resultado no arquivo indicado
			outfile = argv[i+1];
		break;
		case 's':
			// Escalar do parâmetro sigma
			ss = atof(argv[i+1]);
		break;
		}
	}

	// Identificar dispositivo
	init_gpu();

	printf("Dispositivo: \"%s\"\n", deviceProp.name);
	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("CUDA Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	char msg[256];
	sprintf(msg, "Memória Global: %.0f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
	printf("%s", msg);
	printf("(%2d) Multiprocessors, (%3d) CUDA Cores/MP: %d CUDA Cores\n",
		deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

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
	// Soma dos erros das estimativas
	float errsum = 0;

	// Gerar estimativas
	estimar(&train, &estim, ss, &errsum);

	// Exibir erro médio
	printf("Erro médio: %f\n", errsum / (float)estim.total);

	// Salvar resultado no arquivo informado
	if (outfile != NULL ){
		pathSetSave(outfile, &estim);
		printf("Resultado salvo em %s\n", outfile);
	}

	return 0;
}
