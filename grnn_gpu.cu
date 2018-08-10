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

// Obter a quantidade de cuda cores por multiprocessador a partir do Compute Capability
inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        { 0x70, 64 }, // Volta Generation (SM 7.0) GV100 class

        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}

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

	// Cabeçalho do arquivo csv
	printf("Dispositivo\tGeração\tCapacidade\tMultiprocessadores\tCUDA Cores / MP\tMemória Global\tCUDA Driver\tCUDA Runtime\tDimensões da variável independente\tDimensões da variável dependente\tConjunto de treinamento\tConjunto de teste\tErro médio\n");

	// Hardware
	printf("%s\t", deviceProp.name);
	switch ( deviceProp.major ){
		case 3: printf("Kepler"); break;
		case 5: printf("Maxwell"); break;
		case 6: printf("Pascal"); break;
		case 7: printf("Volta"); break;
	}
	printf("\t");
	printf("%d.%d\t", deviceProp.major, deviceProp.minor);
	printf("%d\t", deviceProp.multiProcessorCount);
	printf("%d\t", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
	printf("%.0f MB\t", (float)deviceProp.totalGlobalMem/1048576.0f);
	// Driver e Runtime
	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("%d.%d\t", driverVersion/1000, (driverVersion%100)/10);
	printf("%d.%d\t", runtimeVersion/1000, (runtimeVersion%100)/10);

	struct pathSet train, estim;
	// Carregar arquivo das amostras de treinamento
	pathSetLoad(TRAIN, &train);

	// Arquivo de teste
	pathSetLoad(TEST, &estim);

	printf("%d\t", train.dim[0]);
	printf("%d\t", train.dim[1]);
	printf("%d\t", train.total);
	printf("%d\t", estim.total);

	// Calcular o erro ou salvar um arquivo com o resultado
	// Soma dos erros das estimativas
	float errsum = 0;

	// Gerar estimativas
	estimar(&train, &estim, ss, &errsum);

	// Exibir erro médio
	printf("%f\n", errsum / (float)estim.total);

	// Salvar resultado no arquivo informado
	if (outfile != NULL ){
		pathSetSave(outfile, &estim);
	}

	return 0;
}
