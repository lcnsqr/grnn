// Mapeamento de número inteiro para cor
struct Cor { 
  // Numero de cores disponveis
  int estagios;
  // Pontos de passagem
  int verticesQuant;
  // Intervalos entre vertices
  int intervQuant;
  // Tamanho de cada intervalo
  float intervTaman;
  // Os vertices
  int** vertices;
  // Direcoes (vertices - 1)
  int** direcoes;
};

void corBuild(struct Cor* cor, int estagios){
  cor->estagios = estagios;
  cor->verticesQuant = 7;
  cor->intervQuant = cor->verticesQuant - 1;
  cor->intervTaman = cor->estagios / cor->intervQuant;

  cor->vertices = (int**) malloc(sizeof(int*) * cor->verticesQuant);
  for ( int i = 0; i < cor->verticesQuant; i++ ){
    cor->vertices[i] = (int*) malloc(sizeof(int) * 4);
  }

  cor->direcoes = (int**) malloc(sizeof(int*) * cor->intervQuant);
  for ( int i = 0; i < cor->intervQuant; i++ ){
    cor->direcoes[i] = (int*) malloc(sizeof(int) * 4);
  }

  // Pontos de passagem de cor
  cor->vertices[0][0] = 211;
  cor->vertices[0][1] = 215;
  cor->vertices[0][2] = 207;
  cor->vertices[0][3] = 255;

  cor->vertices[1][0] = 136;
  cor->vertices[1][1] = 138;
  cor->vertices[1][2] = 133;
  cor->vertices[1][3] = 255;

  cor->vertices[2][0] = 114;
  cor->vertices[2][1] = 159;
  cor->vertices[2][2] = 207;
  cor->vertices[2][3] = 255;

  cor->vertices[3][0] = 255;
  cor->vertices[3][1] = 0;
  cor->vertices[3][2] = 0;
  cor->vertices[3][3] = 255;

  cor->vertices[4][0] = 92;
  cor->vertices[4][1] = 53;
  cor->vertices[4][2] = 102;
  cor->vertices[4][3] = 255;

  cor->vertices[5][0] = 115;
  cor->vertices[5][1] = 210;
  cor->vertices[5][2] = 22;
  cor->vertices[5][3] = 255;

  cor->vertices[6][0] = 252;
  cor->vertices[6][1] = 175;
  cor->vertices[6][2] = 62;
  cor->vertices[6][3] = 255;

  // Definir direcoes entre vertices
  for ( int i = 0; i < cor->intervQuant; i++ ){
    for ( int k = 0; k < 4; k++ ){
      cor->direcoes[i][k] = cor->vertices[i + 1][k] - cor->vertices[i][k];
    }
  }
}

void corMap(struct Cor* cor, int c, unsigned int* rgba){
  int k, p;
  k = cor->intervQuant * c / cor->estagios;
  p = c - cor->intervTaman * k;
  for ( int j = 0; j < 4; j++ ){
    rgba[j] = cor->vertices[k][j] + p * cor->direcoes[k][j] / cor->intervTaman;
  }
}
