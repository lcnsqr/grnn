#include <stdio.h>
#include <stdlib.h>
#include "pathio.h"

int main(int argc, char **argv){
	// Converter os dados binários para o formato CSV
	struct pathSet input;
	// Carregar arquivo binário das amostras
	pathSetLoadStdin(&input);
	/*
	printf("Total de amostras: %d\n", input.total);
	printf("Vértices: %d\n", input.vertices);
	for(int i = 0; i < input.vertices; i++){
		printf("Componentes no vértice %d: %d\n", i+1, input.dim[i]);
	}
	// Tipo dos dados
	printf("Tipo dos dados: ");
	switch ( input.type ){
		case 0x01:
			// unsigned char
			printf("char\n");
		break;
		case 0x04:
			// integer
			printf("integer\n");
		break;
		case 0x0f:
			// float
			printf("float\n");
		break;
		case 0x0d:
			// double
			printf("double\n");
		break;
	}
	*/
	// Extrair os pares de variáveis
	for (int i = 0; i < input.total; i++){
		// Variável independente
		for (int d = 0; d < input.dim[0]; d++){
			if ( d == input.dim[0] - 1 ){
				printf("%f", input.data.f[i + d * input.dim[0]]);
			}
			else {
				printf("%f,", input.data.f[i + d * input.dim[0]]);
			}
		}
		printf(" -> ");
		// Variável dependente
		for (int d = 0; d < input.dim[1]; d++){
			if ( d == input.dim[1] - 1 ){
				printf("%f", input.data.f[input.total * input.dim[0] + i + d * input.dim[1]]);
			}
			else {
				printf("%f,", input.data.f[input.total * input.dim[0] + i + d * input.dim[1]]);
			}
		}
		printf("\n");
	}
}
