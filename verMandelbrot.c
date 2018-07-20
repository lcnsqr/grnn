#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "SDL.h"
#include "color.h"
#include "pathio.h"

// Parâmetros iniciais
#define XMIN -2.5
#define XMAX 1
#define YMIN -1.75
#define YMAX 1.75
#define ITERATIONS 1000
#define WIDTH 400
#define HEIGHT 400
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
	int iterations;
	// Imagem
	struct View view;
	// Exibição e interação por SDL
	SDL_Window *window;
	SDL_Renderer *renderer;
	SDL_Texture *texture;
	SDL_Event event;
	// Conjunto do resultado
	struct pathSet result;
};

void finalizar(struct Context *ctx){
	free(ctx->view.frame);
	// Encerrar SDL
	SDL_DestroyTexture(ctx->texture);
	SDL_DestroyRenderer(ctx->renderer);
	SDL_DestroyWindow(ctx->window);
	SDL_Quit();
}

void render(struct Context *ctx){
	float x,y;
	// Cor
	unsigned int c, rgba[4];
	#pragma omp parallel for private(x,y,c,rgba)
	for (int i = 0; i < ctx->view.frameSize; i += ctx->view.depth){
		// Conjunto do resultado
		x = ctx->result.data.f[3*(i/ctx->view.depth)];
		y = ctx->result.data.f[3*(i/ctx->view.depth)+1];
		c = floor(ctx->iterations * ctx->result.data.f[3*(i/ctx->view.depth)+2]);
		// Pixels
		corMap(&ctx->view.cor, c, rgba);
		ctx->view.frame[i] = (char)rgba[0];
		ctx->view.frame[i+1] = (char)rgba[1];
		ctx->view.frame[i+2] = (char)rgba[2];
		ctx->view.frame[i+3] = (char)rgba[3];
	}
	
	// Atualizar exibição
	SDL_UpdateTexture(ctx->texture, NULL, ctx->view.frame, DEPTH * ctx->view.width * sizeof(char));
	SDL_RenderClear(ctx->renderer);
	SDL_RenderCopy(ctx->renderer, ctx->texture, NULL, NULL);

	SDL_RenderPresent(ctx->renderer);
}

int main(int argc, char **argv){
	// Contextos de renderização
	struct Context ctx;
	ctx.iterations = ITERATIONS;
	// Estrutura da imagem
	ctx.view.width = WIDTH;
	ctx.view.height = HEIGHT;
	ctx.view.depth = DEPTH;
	ctx.view.frameSize = ctx.view.width*ctx.view.height*ctx.view.depth;
	ctx.view.frame = (char *)malloc(sizeof(char)*ctx.view.frameSize);
	// Total de cores equivale ao número de iterações
	corBuild(&ctx.view.cor, ctx.iterations);

	// Verificar se o nome do arquivo foi informado
	if ( argc < 1 ){
		puts("Um nome de arquivo com resultados estimados deve ser informado como argumento para o comando");
		exit(EXIT_FAILURE);
	}
	// Carregar arquivo da estimativa
	pathSetLoad(argv[1], &ctx.result);
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Erro ao iniciar o SDL: %s", SDL_GetError());
		exit(-1);
	}

	// Mandelbrot
	if (SDL_CreateWindowAndRenderer(ctx.view.width, ctx.view.height, 0, &ctx.window, &ctx.renderer)) {
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Impossível criar a janela e o renderizador: %s", SDL_GetError());
		exit(-1);
	}
	SDL_SetWindowTitle(ctx.window, "Mandelbrot Estimado");
	ctx.texture = SDL_CreateTexture(ctx.renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, ctx.view.width, ctx.view.height);

	render(&ctx);

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
		SDL_PumpEvents();
		/*
		if (SDL_GetMouseState(&ctx.curPos, &ctx.curPos[1]) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
			// Ação do mouse 
		}
		*/
	}
}
