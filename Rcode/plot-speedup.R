library("ggplot2")
setwd("~/GIT/grnn/")

problem_size <- c("300x300", "700x700", "1000x1000")
n_threads <- 1:40
app <- "Mandelbrot"
df <- data.frame()    
for (N in problem_size){
    for (thread in n_threads){
        temp_var <- t(read.csv(paste("./data/CPU/mandelbrot/CPU_", thread, "_", N, ".dat", sep=""),sep = "\t")["Tempo.gasto"])
        if(thread == 1){
            baseM = mean(temp_var)
            baseSD = sd(temp_var)
        }
        
    meanFile <- mean(temp_var)
    sdFile <- sd(temp_var)
    speedup <- baseM/meanFile
    speedupSD <- sd(baseM/temp_var)
    df <- rbind(df, cbind(meanFile, sdFile, speedup, speedupSD, N, thread, app))
    }
}

df$meanFile <- as.numeric(as.character(df$meanFile))
df$speedup <- as.numeric(as.character(df$speedup))
df$speedupSD <- as.numeric(as.character(df$speedupSD))

df$LCL <- df$speedup - df$speedupSD
df$UCL <- df$speedup + df$speedupSD

# mode(df$)
colnames(df) <- c("meanFile", "sdFile", "speedup", "speedupSD", "N", "thread", "app", "LCL", "UCL")


Graph <- ggplot(df, aes(x=thread, y=speedup, group=N, color=N))  +
    geom_line(size=1.25) + 
    geom_point(size=1.25) +
    geom_errorbar(aes(ymin=LCL, ymax=UCL), size=1.25) +
    xlab("Número de Threads")+
    ylab("SpeedUp (Base=Sequencial)") +
    theme_bw() +
    # scale_colour_manual(values=c(cbbPalette, "blue")) +
    # scale_fill_manual(values=c(cbbPalette, "blue")) +
    ggtitle(paste("SpeedUp em CPU de ", app, " com 40 Threads", sep = "")) +
    theme(plot.title = element_text(family = "Times", face="bold", size=25, colour = "Black")) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.title = element_text(family = "Times", face="bold", size=20, colour = "Black")) +
    theme(axis.text  = element_text(family = "Times", face="bold", size=14, colour = "Black")) +
    theme(legend.title  = element_text(family = "Times", face="bold", size=0, colour = "Black")) +
    theme(legend.title.align=0.5) +
    theme(legend.text  = element_text(family = "Times", face="bold", size=20, colour = "Black")) +
    theme(legend.key.size = unit(5, "cm")) +
    theme(legend.direction = "horizontal",
          legend.position = "bottom",
    legend.key=element_rect(size=0),
    legend.key.size = unit(2, "lines")) +
    guides(col = guide_legend(nrow = 1))

ggsave(paste("./images/CPU-", app, ".pdf",sep=""), Graph, height=16, width=24, units="cm")


##### DIFUSION PROBLEM

problem_size <- c(2^17, 2^18, 2^19)
n_threads <- 1:40
app <- "Difusão"
df <- data.frame()    
for (N in problem_size){
    for (thread in n_threads){
        temp_var <- t(read.csv(paste("./data/CPU/difusao/CPU_", thread, "_", N, ".dat", sep=""),sep = "\t")["Tempo.gasto"])
        if(thread == 1){
            baseM = mean(temp_var)
            baseSD = sd(temp_var)
        }
        
        meanFile <- mean(temp_var)
        sdFile <- sd(temp_var)
        speedup <- baseM/meanFile
        speedupSD <- sd(baseM/temp_var)
        df <- rbind(df, cbind(meanFile, sdFile, speedup, speedupSD, N, thread, app))
    }
}

df$meanFile <- as.numeric(as.character(df$meanFile))
df$speedup <- as.numeric(as.character(df$speedup))
df$speedupSD <- as.numeric(as.character(df$speedupSD))

df$LCL <- df$speedup - df$speedupSD
df$UCL <- df$speedup + df$speedupSD

# mode(df$)
colnames(df) <- c("meanFile", "sdFile", "speedup", "speedupSD", "N", "thread", "app", "LCL", "UCL")


Graph <- ggplot(df, aes(x=thread, y=speedup, group=N, color=N))  +
    geom_line(size=1.25) + 
    geom_point(size=1.25) +
    geom_errorbar(aes(ymin=LCL, ymax=UCL), size=1.25) +
    xlab("Número de Threads")+
    ylab("SpeedUp (Base=Sequencial)") +
    theme_bw() +
    # scale_colour_manual(values=c(cbbPalette, "blue")) +
    # scale_fill_manual(values=c(cbbPalette, "blue")) +
    ggtitle(paste("SpeedUp em CPU de ", app, " com 40 Threads", sep = "")) +
    theme(plot.title = element_text(family = "Times", face="bold", size=25, colour = "Black")) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.title = element_text(family = "Times", face="bold", size=20, colour = "Black")) +
    theme(axis.text  = element_text(family = "Times", face="bold", size=14, colour = "Black")) +
    theme(legend.title  = element_text(family = "Times", face="bold", size=0, colour = "Black")) +
    theme(legend.title.align=0.5) +
    theme(legend.text  = element_text(family = "Times", face="bold", size=20, colour = "Black")) +
    theme(legend.key.size = unit(5, "cm")) +
    theme(legend.direction = "horizontal",
          legend.position = "bottom",
          legend.key=element_rect(size=0),
          legend.key.size = unit(2, "lines")) +
    guides(col = guide_legend(nrow = 1))

ggsave(paste("./images/CPU-", app, ".pdf",sep=""), Graph, height=16, width=24, units="cm")