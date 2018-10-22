#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "SDL.h"
#include "color.h"
#include "pathio.h"

// Valor uniformemente distribuído em [0,1)
#define RAND ((double)(rand() >> 1)/((RAND_MAX >> 1) + 1))

// Tamanho do conjunto de teste
#define TESTSIZE 8192

// Parâmetros iniciais
#define ITERATIONS 1000
#define DEPTH 4

struct View {
	int width, height, depth;
	int frameSize;
	char *frame;
	// Estrutura de mapeamento posição -> cor
	struct Cor cor;
};

// Contexto de execução 
struct Context {
	// Parâmetros do conjunto mandelbrot
	float xmin, xmax, ymin, ymax;
	int iterations;
	// Imagem
	struct View view;
	// Exibição e interação por SDL
	SDL_Window *window;
	SDL_Renderer *renderer;
	SDL_Texture *texture;
	SDL_Event event;
	// Posição do cursor relativo à janela
	int curPos[2];
	// Conjunto de treinamento
	struct pathSet train;
	// Conjunto de teste
	struct pathSet test;
};

void finalizar(struct Context *ctx){
	free(ctx->view.frame);
	// Encerrar SDL
	SDL_DestroyTexture(ctx->texture);
	SDL_DestroyRenderer(ctx->renderer);
	SDL_DestroyWindow(ctx->window);
	SDL_Quit();
}

double mandelIter(double cx, double cy, int iterations){
	float x = 0, y = 0;
	float xx = 0, yy = 0, xy = 0;
	int i = 0;
	while (i < iterations && xx + yy <= 4){
		xy = x * y;
		xx = x * x;
		yy = y * y;
		x = xx - yy + cx;
		y = xy + xy + cy;
		i++;
	}
	return (float)i;
}

void render(struct Context *ctx, int window){
	// Coordenadas na imagem
	int ix, iy;
	// Coordenadas do ponto candidato
	float x, y;
	// Cor
	int c;
	unsigned int rgba[4];
	#pragma omp parallel for private(ix,iy,x,y,c,rgba)
	for (int i = 0; i < ctx->view.frameSize; i += ctx->view.depth){
		ix = (i/ctx->view.depth) % ctx->view.width;
		iy = (i/ctx->view.depth) / ctx->view.width;
		// A orientação vertical é invertida na imagem
		iy = ctx->view.height - iy - 1;
    x = ctx->xmin + (ctx->xmax - ctx->xmin) * ix / (ctx->view.width - 1);
    y = ctx->ymin + (ctx->ymax - ctx->ymin) * iy / (ctx->view.height - 1);
		c = floor((float)ITERATIONS * log(mandelIter(x, y, ctx->iterations) )/log(ITERATIONS+1));
		// Conjunto de treinamento
		ctx->train.data.f[ (i/ctx->view.depth) ] = x;
		ctx->train.data.f[ (i/ctx->view.depth) + ctx->train.total ] = y;
		ctx->train.data.f[ (i/ctx->view.depth) + ctx->train.total*2 ] = (float)c/ITERATIONS;
		// Pixels
		corMap(&ctx->view.cor, c, rgba);
		ctx->view.frame[i] = (char)rgba[0];
		ctx->view.frame[i+1] = (char)rgba[1];
		ctx->view.frame[i+2] = (char)rgba[2];
		ctx->view.frame[i+3] = (char)rgba[3];
	}
	
  if ( window == 1 ){
    // Atualizar exibição
    SDL_UpdateTexture(ctx->texture, NULL, ctx->view.frame, DEPTH * ctx->view.width * sizeof(char));
    SDL_RenderClear(ctx->renderer);
    SDL_RenderCopy(ctx->renderer, ctx->texture, NULL, NULL);
    SDL_RenderPresent(ctx->renderer);
  }
}

int main(int argc, char **argv){
	// Opções da linha de comando
  float XMIN = -2.5;
  float XMAX = 1;
  float YMIN = -1.75;
  float YMAX = 1.75;
	unsigned int WIDTH = 700;
	unsigned int HEIGHT = 700;
  // Exibir tela interativa
  int window = 1;
	for(int i = 1; i < argc; i++){
		switch (argv[i][1]){
		case 'w':
			// Largura
      WIDTH = atoi(argv[i+1]);
		break;
		case 'h':
			// Altura
      HEIGHT = atoi(argv[i+1]);
		break;
		case 't':
      // Top
      YMAX = atof(argv[i+1]);
		break;
		case 'r':
      // Right
      XMAX = atof(argv[i+1]);
		break;
		case 'b':
      // Bottom
      YMIN = atof(argv[i+1]);
		break;
		case 'l':
      // Left
      XMIN = atof(argv[i+1]);
		break;
		case 's':
      // Não exibir janela interativa
      window = 0;
		break;
		}
	}
	// Semente aleatória
	srand((unsigned int)time(NULL));
	// Contextos de renderização
	struct Context ctx;
	ctx.xmin = XMIN;
	ctx.xmax = XMAX;
	ctx.ymin = YMIN;
	ctx.ymax = YMAX;
	ctx.iterations = ITERATIONS;
	// Estrutura da imagem
	ctx.view.width = WIDTH;
	ctx.view.height = HEIGHT;
	ctx.view.depth = DEPTH;
	ctx.view.frameSize = ctx.view.width*ctx.view.height*ctx.view.depth;
	ctx.view.frame = (char *)malloc(sizeof(char)*ctx.view.frameSize);
	// Total de cores equivale ao números de iterações
	corBuild(&ctx.view.cor, ctx.iterations);

	// Conjunto de treinamento
	ctx.train.type = 0x0f;
	ctx.train.total = WIDTH * HEIGHT;
	ctx.train.vertices = 2;
	ctx.train.dim = (unsigned int*)malloc(2*sizeof(unsigned int));
	ctx.train.dim[0] = 2;
	ctx.train.dim[1] = 1;
	ctx.train.size = ctx.train.total * 3 * sizeof(float);
	ctx.train.data.f = (float*)malloc(ctx.train.size);
	// Conjunto de teste
	ctx.test.type = 0x0f;
	ctx.test.total = TESTSIZE;
	ctx.test.vertices = 2;
	ctx.test.dim = (unsigned int*)malloc(2*sizeof(unsigned int));
	ctx.test.dim[0] = 2;
	ctx.test.dim[1] = 1;
	ctx.test.size = ctx.test.total * 3 * sizeof(float);
	ctx.test.data.f = (float*)malloc(ctx.test.size);
  // Variável (auxiliares) independente no conjunto de teste
  float x, y;
  // Variável (auxiliar) dependente no conjunto de teste
  int c;

  if ( window == 0 ){
    // Modo não interativo
    render(&ctx, window);
    // Gerar conjunto de treinamento
    pathSetSave("train.bin", &ctx.train);
    // Produzir pontos aleatórios na área correspondente ao conjunto 
    // de treinamento para servirem como conjunto de teste
    #pragma omp parallel for private(x,y,c)
    for (int i = 0; i < ctx.test.total; i++){
      x = ctx.xmin + (ctx.xmax - ctx.xmin) * RAND;
      y = ctx.ymin + (ctx.ymax - ctx.ymin) * RAND;
      c = floor((float)ITERATIONS * log(mandelIter(x, y, ctx.iterations) )/log(ITERATIONS+1));
      // Conjunto de teste
      ctx.test.data.f[ i ] = x;
      ctx.test.data.f[ i + ctx.test.total ] = y;
      ctx.test.data.f[ i + ctx.test.total*2 ] = (float)c/ITERATIONS;
    }
    pathSetSave("test.bin", &ctx.test);
		exit(0);
  }

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Erro ao iniciar o SDL: %s", SDL_GetError());
		exit(-1);
	}

	// Mandelbrot
	if (SDL_CreateWindowAndRenderer(ctx.view.width, ctx.view.height, 0, &ctx.window, &ctx.renderer)) {
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Impossível criar a janela e o renderizador: %s", SDL_GetError());
		exit(-1);
	}
	SDL_SetWindowTitle(ctx.window, "Mandelbrot");
	ctx.texture = SDL_CreateTexture(ctx.renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, ctx.view.width, ctx.view.height);

  // Gerar fractal
	render(&ctx, window);

	// Recolher eventos
	while (1){
		SDL_PollEvent(&ctx.event);
		if (ctx.event.type == SDL_QUIT){
			finalizar(&ctx);
			return 0;
		}
		else if (ctx.event.type == SDL_WINDOWEVENT){
			switch (ctx.event.window.event){
			case SDL_WINDOWEVENT_MOVED:
			case SDL_WINDOWEVENT_FOCUS_GAINED:
			case SDL_WINDOWEVENT_RESTORED:
				SDL_RenderPresent(ctx.renderer);
			break;
			}
		}
		else if( ctx.event.type == SDL_KEYDOWN ){
			if ( ctx.event.key.keysym.sym == SDLK_q ){ 
				// Tecla "q" encerra
				finalizar(&ctx);
				return 0;
			}
			else if ( ctx.event.key.keysym.sym == SDLK_UP ){ 
				// Seta pra cima
				float dy = (ctx.ymax - ctx.ymin)*1e-1;
				ctx.ymin += dy;
				ctx.ymax += dy;
				render(&ctx, window);
			}
			else if ( ctx.event.key.keysym.sym == SDLK_DOWN ){ 
				// Seta pra baixo
				float dy = (ctx.ymax - ctx.ymin)*1e-1;
				ctx.ymin -= dy;
				ctx.ymax -= dy;
				render(&ctx, window);
			}
			else if ( ctx.event.key.keysym.sym == SDLK_LEFT ){ 
				// Seta pra cima
				float dx = (ctx.xmax - ctx.xmin)*1e-1;
				ctx.xmin -= dx;
				ctx.xmax -= dx;
				render(&ctx, window);
			}
			else if ( ctx.event.key.keysym.sym == SDLK_RIGHT ){ 
				// Seta pra baixo
				float dx = (ctx.xmax - ctx.xmin)*1e-1;
				ctx.xmin += dx;
				ctx.xmax += dx;
				render(&ctx, window);
			}
			else if ( ctx.event.key.keysym.sym == SDLK_z ){ 
				// zoom in
				float mx = (ctx.xmin + ctx.xmax)/2.0;
				float my = (ctx.ymin + ctx.ymax)/2.0;
				float dx = (ctx.xmax - ctx.xmin)/4.0;
				float dy = (ctx.ymax - ctx.ymin)/4.0;
				ctx.xmin = mx - dx;
				ctx.xmax = mx + dx;
				ctx.ymin = my - dy;
				ctx.ymax = my + dy;
				render(&ctx, window);
			}
			else if ( ctx.event.key.keysym.sym == SDLK_x ){ 
				// zoom out
				float mx = (ctx.xmin + ctx.xmax)/2.0;
				float my = (ctx.ymin + ctx.ymax)/2.0;
				float dx = (ctx.xmax - ctx.xmin);
				float dy = (ctx.ymax - ctx.ymin);
				ctx.xmin = mx - dx;
				ctx.xmax = mx + dx;
				ctx.ymin = my - dy;
				ctx.ymax = my + dy;
				render(&ctx, window);
			}
			else if ( ctx.event.key.keysym.sym == SDLK_s ){ 
				// Gravar conjunto de treinamento
				pathSetSave("train.bin", &ctx.train);
				printf("Conjunto de treinamento train.bin salvo\n");
        // Produzir pontos aleatórios na área correspondente ao conjunto 
        // de treinamento para servirem como conjunto de teste
        #pragma omp parallel for private(x,y,c)
        for (int i = 0; i < ctx.test.total; i++){
          x = ctx.xmin + (ctx.xmax - ctx.xmin) * RAND;
          y = ctx.ymin + (ctx.ymax - ctx.ymin) * RAND;
          c = floor((float)ITERATIONS * log(mandelIter(x, y, ctx.iterations) )/log(ITERATIONS+1));
          // Conjunto de teste
          ctx.test.data.f[ i ] = x;
          ctx.test.data.f[ i + ctx.test.total ] = y;
          ctx.test.data.f[ i + ctx.test.total*2 ] = (float)c/ITERATIONS;
        }
				pathSetSave("test.bin", &ctx.test);
				printf("Conjunto de teste test.bin salvo\n");
			}
			else if ( ctx.event.key.keysym.sym == SDLK_i ){ 
        // Exibir informações
        fprintf(stderr, "Limites:\n\tEsquerdo:\t%f\n\tDireito:\t%f\n\tInferior:\t%f\n\tSuperior:\t%f\n", ctx.xmin, ctx.xmax, ctx.ymin, ctx.ymax);
        fprintf(stderr, "Resolução:\n\t%f pontos por unidade\n", (float)WIDTH / fabs(ctx.xmax - ctx.xmin));
			}
		}
		SDL_PumpEvents();
		/*
		if (SDL_GetMouseState(&ctx.curPos, &ctx.curPos[1]) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
			// Ação do mouse 
		}
		*/
	}
}
