require(ggplot2)
require(dplyr)
require(tidyr)
require(ggpubr)
require(cowplot)

# load in latent components ------
grp.comp.1 <- read.csv('grp_comp_1.csv')
grp.comp.2 <- read.csv('grp_comp_2.csv')
grp.comp.3 <- read.csv('grp_comp_3.csv')

dim.comp.1 <- read.csv('dim_comp_1.csv')
dim.comp.2 <- read.csv('dim_comp_2.csv')
dim.comp.3 <- read.csv('dim_comp_3.csv')

# read in infer_unobserved
df <- read.csv('infer_unobserved.csv')
set_state <- df %>% dplyr::group_by(state1) %>% summarize(set = mean(set))

# re-numbering the components to match across models based on highest correlation (very hacky)-----
allval <- cbind(dim.comp.1$value,dim.comp.2$value,dim.comp.3$value, 
                grp.comp.1$value,grp.comp.2$value,grp.comp.3$value)
allcor <- allval %>% as.data.frame() %>% drop_na() %>% cor() %>% as.data.frame()
colnames(allcor) <- c('dim.comp.1','dim.comp.2','dim.comp.3',
                      'grp.comp.1','grp.comp.2','grp.comp.3')
rownames(allcor) <- colnames(allcor)
crosscor <-  allcor[1:3,4:6] %>% as.data.frame()
# find component 1
maxind <- which(crosscor == max(crosscor),arr.ind = T)
dim.comp.1_ <- get(rownames(crosscor)[maxind[1]])
grp.comp.1_ <- get(colnames(crosscor)[maxind[2]])
crosscor <- crosscor[-maxind[1],-maxind[2]]
# find component 2
maxind <- which(crosscor == max(crosscor),arr.ind = T)
dim.comp.2_ <- get(rownames(crosscor)[maxind[1]])
grp.comp.2_ <- get(colnames(crosscor)[maxind[2]])
# find component 3
thirdind <- 3 - maxind
dim.comp.3_ <- get(rownames(crosscor)[thirdind[1]])
grp.comp.3_ <- get(colnames(crosscor)[thirdind[2]]) 

grp.comp.1 <- grp.comp.1_
grp.comp.2 <- grp.comp.2_
grp.comp.3 <- grp.comp.3_
dim.comp.1 <- dim.comp.1_
dim.comp.2 <- dim.comp.2_
dim.comp.3 <- dim.comp.3_

# function to draw heatmap from long-format predicted transition prob input ------
component.heatmap <- function(comp){
  g1 <- ggplot(data = comp %>% filter(set == 1)) + 
    geom_tile(aes(x = state2,y = state1, fill = value)) +
    scale_fill_gradientn(limits = c(0,1),colours=c("navyblue", "white", "darkorange1")) +
    theme(axis.title.x = element_blank()) +
    ylab('From')
  
  g2 <- ggplot(data = comp %>% filter(set == 2)) + 
    geom_tile(aes(x = state2,y = state1, fill = value)) +
    scale_fill_gradientn(limits = c(0,1),colours=c("navyblue", "white", "darkorange1")) +
    theme(axis.title.x = element_blank())  +
    ylab('From') 
  
  g3 <- ggplot(data = comp %>% filter(set == 3)) + 
    geom_tile(aes(x = state2,y = state1, fill = value)) +
    scale_fill_gradientn(limits = c(0,1),colours=c("navyblue", "white", "darkorange1")) +
    ylab('From') + xlab('To')
  
  ga <- plot_grid(g1,g2,g3,ncol = 1,align = 'v')
  
  return(ga)
}


g.dim.comp.1 <- component.heatmap(dim.comp.1)
g.dim.comp.2 <- component.heatmap(dim.comp.2)
g.dim.comp.3 <- component.heatmap(dim.comp.3)

g.grp.comp.1 <- component.heatmap(grp.comp.1)
g.grp.comp.2 <- component.heatmap(grp.comp.2)
g.grp.comp.3 <- component.heatmap(grp.comp.3)

# function to generate classification/mixture heatmaps --------
# function for mixture heatmap, one set (nolegend)
gmix.oneset <- function(subset){
  
  g <- ggplot(data = subset) + 
    geom_tile(aes(x = state2,y = state1, fill = value)) +
    scale_fill_gradientn(limits = c(0,1),colours=c("navyblue", "white", "darkorange1")) +
    theme(axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          legend.position = 'none')
  
  return(g)
}

# hard group heatmap
grp.h.heatmap <- function(from, to, tp){
  temp <- df[(df$state1 == from) & (df$state2 == to) & (df$step == tp*10-1),]
  cname <- paste('grp.comp.',temp$grp_h + 1,sep = '')
  comp <- get(cname)
  g1 <- gmix.oneset(comp %>% filter(set == 1))
  g2 <- gmix.oneset(comp %>% filter(set == 2))
  g3 <- gmix.oneset(comp %>% filter(set == 3))
  return(plot_grid(g1,g2,g3,align = 'v',ncol = 1))
}

grp.s.heatmap <- function(from, to, tp){
  temp <- df[(df$state1 == from) & (df$state2 == to) & (df$step == tp*10-1),]
  mix <- grp.comp.1
  mix$value <- temp$grp_s_1 * grp.comp.1$value +
    temp$grp_s_2 * grp.comp.2$value +
    temp$grp_s_3 * grp.comp.3$value 
  g1 <- gmix.oneset(mix %>% filter(set == 1))
  g2 <- gmix.oneset(mix %>% filter(set == 2))
  g3 <- gmix.oneset(mix %>% filter(set == 3))
  return(plot_grid(g1,g2,g3,align = 'v',ncol = 1))
}

dim.heatmap <- function(from, to, tp){
  temp <- df[(df$state1 == from) & (df$state2 == to) & (df$step == tp*10-1),]
  mix <- dim.comp.1
  mix$value <- temp$dim_1 * dim.comp.1$value +
    temp$dim_2 * dim.comp.2$value +
    temp$dim_3 * dim.comp.3$value 
  g1 <- gmix.oneset(mix %>% filter(set == 1))
  g2 <- gmix.oneset(mix %>% filter(set == 2))
  g3 <- gmix.oneset(mix %>% filter(set == 3))
  return(plot_grid(g1,g2,g3,align = 'v',ncol = 1))
}



