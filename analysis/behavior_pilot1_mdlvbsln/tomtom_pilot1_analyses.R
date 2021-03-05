require(dplyr)
require(stringr)
require(tidyr)
require(reshape2)
require(blme)
require(lmerTest)
require(Rmisc)
require(ggplot2)

# define standardized beta function
stdCoef.merMod <- function(object) {
  sdy <- sd(getME(object,"y"))
  sdx <- apply(getME(object,"X"), 2, sd)
  sc <- fixef(object)*sdx/sdy
  se.fixef <- coef(summary(object))[,"Std. Error"]
  se <- se.fixef*sdx/sdy
  return(data.frame(stdcoef=sc, stdse=se))
}


# get stimuli level characteristics, mostly the predicted direction of change
pair_sign <- read.csv('pair_sign.csv')
pair_sign$sign <- rowMeans(pair_sign[c("change_grp_hard","change_dim" )]) > 0
# load in raw model inferece (rather than change scores)
# reverse coding of the low condition was already done in python
df.rawinf <- read.csv('inference_hilo.csv')
df.rawinf <- df.rawinf %>%
  select(!X) %>%
  melt(
    id.vars ='p_pair',
    measure.vars = colnames(.)[!grepl('pair',colnames(.))]
  ) %>% 
  tidyr::separate(col = 'variable', into = c('model','cond'), sep = '_') %>% # split up the condition strings
  # dcast(p_pair + cond ~ model) %>%
  dplyr::rename(states = p_pair, inference = value)

# pair_sign <- pair_sign %>% select(!c('change_grp_hard','change_dim'))

# read data
df <- read.csv('tomtom_pilot1_full.csv')

# prolific check-----
#combine
prolus = read.csv('tomtom_pilot1_2_prolificreport.csv')
prolus <- prolus %>% dplyr::rename(PID = participant_id)
# df <- left_join(prolus,df,by='PID')
# 
# # find responses that might not meet requirement
# dfproblem <- df %>%
#   filter(
#     Progress < 97 | entered_code != '3D09A4A1'
#   )

# write.csv(dfproblem,'prolific_check.csv')

# actual analyses-----
# add pair columns to data
# rename_fn <- function(cols){ # custom name change function
#   newcols <- rep(NA,length(cols))
#   for(i in 1:length(cols)){
#     col <- cols[i]
#     if(!grepl('b1|b2',col)){
#       newcol <- paste(substring(col,1,1), substring(col,2),sep = '_')
#     } else {
#       newcol <- paste(substring(col,1,2), substring(col,3),sep = '_')
#     }
#     newcols[i] <- tolower(newcol)
#   }
#   return(newcols)
# } 

#optional filtering based on total time spent, does not affect results
med <- fivenum(df$Duration..in.seconds.)[3]
iqi <- fivenum(df$Duration..in.seconds.)[4] - fivenum(df$Duration..in.seconds.)[2]
df <- df %>% filter(.['Duration..in.seconds.'] >= med - 1.5*iqi,
                    .['Duration..in.seconds.'] <= med + 1.5*iqi)

df <- df %>%
        filter(
          Progress >= 97, # only complete responses
          attn2 == 2 # filter by comprehension check, maybe add attn 1 as well
        ) %>% 
        # rename_with(rename_fn,ends_with('From')|ends_with('To')) %>% #rename columsn prep for melt
        mutate(
          high_states = paste(hPredFrom,hPredTo,sep='-'), #Stim not included, preds uniquely identify
          low_states = paste(lPredFrom,lPredTo,sep='-'),
          base1_states = paste(b1PredFrom,b1PredTo,sep='-'),
          base2_states = paste(b2PredFrom,b2PredTo,sep='-'),
        ) %>%
        select( # for now only extract essential columns
          PID,high_1,low_1,base1_1,base2_1,ends_with('states')
        )
        
colnames(df) <- str_remove(colnames(df),'_1')
  
cns <- colnames(df)
df_left <- melt(df,id.vars = 'PID',measure.vars = cns[!grepl('PID|states',cns)],
                value.name = 'rating',variable.name = 'cond')
df_right <- melt(df,id.vars = 'PID',measure.vars = cns[grepl('states',cns)],
                 value.name = 'states',variable.name = 'cond')
df_right$cond <- str_remove(df_right$cond,'_states')
df <- left_join(
  x = df_left,
  y = df_right,
  by = c('PID','cond')
)

#flip the high and low for the "false states" (pred should be neg cor with stim)
flip_states <- pair_sign$p_pair[!pair_sign$sign]
high_flip_ind <- which((df$cond == 'high') & (df$states %in% flip_states))
low_flip_ind <- which((df$cond == 'low') & (df$states %in% flip_states))
df$cond[high_flip_ind] <- 'low'
df$cond[low_flip_ind] <- 'high'


# actual analysis ----------
# the essential test, comparing high vers low
set.seed(13621)
mdl.hilo <- lmer(data = df[!grepl('base',df$cond),], rating ~ cond + (1+cond|states) + (1|PID))

# test including the baseline conditions
df$cond[grep('base',df$cond)] <- 'base'
df$cond <- factor(df$cond, levels = c('base','low','high'))
set.seed(54545)
mdl.allcond <- lmer(data = df, rating ~ cond + (1+cond|states) + (1|PID))

# some basic visualization
dfse <- df %>% summarySEwithin(measurevar = 'rating',withinvars = 'cond',idvar = 'PID')
viz.cond <- ggplot() +
  geom_boxplot(data = df, aes(x = cond, y = rating, color = cond)) +
  geom_point(data = dfse, aes(x = cond, y = rating, color = cond)) +
  geom_errorbar(data = dfse, aes(x = cond, ymin = rating-ci,ymax = rating+ci, color = cond),width = .3)

# same analyses but restricted to the american sample
dfus <- df %>% filter(PID %in% prolus$PID)
set.seed(126312)
mdl.hilo.us <- lmer(data = dfus[!grepl('base',dfus$cond),], rating ~ cond + (1+cond|states) + (1|PID))
stdCoef.merMod(mdl.hilo.us)
set.seed(642673)
mdl.allcond.us <- lmer(data = dfus, rating ~ cond + (1+cond|states) + (1+cond|PID))

# some basic visualization
dfusse <- dfus %>% summarySEwithin(measurevar = 'rating',withinvars = 'cond',idvar = 'PID')
viz.cond.us <- ggplot() +
  geom_boxplot(data = dfus, aes(x = cond, y = rating, color = cond)) +
  geom_point(data = dfusse, aes(x = cond, y = rating, color = cond)) +
  geom_errorbar(data = dfusse, aes(x = cond, ymin = rating-ci,ymax = rating+ci, color = cond),width = .3)
# a version that also also plots model prediction in each condition
# first add model prediction to the df of means
dfusse <- df.rawinf %>% 
            dcast(states+cond~model,value.var = 'inference') %>% 
            group_by(cond) %>%
            dplyr::summarize(grp = mean(grp)*100,dim = mean(dim)*100) %>%
            right_join(dfusse, by = 'cond') 

viz.cond.us.wmdl <- ggplot() +
  geom_boxplot(data = dfus, aes(x = cond, y = rating, color = cond)) +
  geom_errorbar(data = dfusse, aes(x = cond, ymin = rating-ci,ymax = rating+ci, color = cond),width = .3) +
  geom_point(data = dfusse %>%
               select(cond, grp, dim, rating) %>%
               melt(id.vars = 'cond') %>%
               mutate(variable = factor(variable, 
                                        levels = c('rating','grp','dim'),
                                        labels = c('Human','Group','Dimensional'))), 
             aes(x = cond, y = value, color = cond, shape = variable), size = 3) +
  labs(x = 'Condition',y = 'Transition Probability',
       color = 'Condition',shape = 'Source') +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"),
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 12,color= 'black'),
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 12,color= 'black'))
# ggsave('tomtom_pilot1_means.png',plot = viz.cond.us.wmdl,width = 6,height = 6)
  
# more fine grained comparison against the models
# hi-lo diff by state
fixef.us <- fixef(mdl.allcond.us)
ranef.us <- as.data.frame(ranef(mdl.allcond.us)$states)
colnames(ranef.us) <- c('base','low','high')
ranef.us <- ranef.us %>%
  mutate(base = base + fixef.us[1]) %>% # get each state's estimated baseline
  mutate(
    low = low + base + fixef.us[2], # get each state's estimated high low
    high = high + base + fixef.us[3]
  ) %>%
  mutate( # compute "change scores"
    change = high - low
  ) %>%
  tibble::rownames_to_column('states') # states are coded as row index, convert to column

dfchange <- pair_sign %>% 
  select('p_pair',starts_with('change')) %>%
  dplyr::rename(states = p_pair) %>%
  left_join(ranef.us, by = 'states')

cor(abs(dfchange$change_grp_hard),dfchange$change)
cor(abs(dfchange$change_dim),dfchange$change)
plot(abs(dfchange$change_dim),dfchange$change)
plot(abs(dfchange$change_grp_hard),dfchange$change)
viz.hilodiff.corwmdl <- ggplot(
    data = dfchange %>%
      select(states, starts_with('change')) %>%
      mutate(group = abs(change_grp_hard), dim = abs(change_dim), human = change)%>%
      select(!starts_with('change')) %>%
      melt(id.vars = c('states','human'),variable.name = 'model',value.name = 'inference'),
    aes(x = inference, y = human, color = model)
  ) + 
  geom_point() + geom_smooth(method = 'lm',se = F)+
  labs(x = 'Model Prediction',y = 'Human Judgment', color = 'Condition') +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"),
        axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 12,color= 'black'),
        axis.title.y = element_text(size = 15),
        axis.text.y = element_text(size = 12,color= 'black'))
# ggsave('tomtom_pilot1_corr_hmnmdlchange.png',plot = viz.cond.us.wmdl,width = 6,height = 6)

# do the "model comparison", at the rating level
#join behavior data frame and model data frame
dfhvm <- left_join(dfus, df.rawinf, by = c('states','cond')) # human vs machine
dfhvm <- dfhvm %>%
  mutate(
    rating = scale(rating), 
    inference = scale(inference),
    model = factor(model,levels = c('grp','dim'))
  )
mdl.hvm <- lmer(data = dfhvm %>% filter(cond != 'base'), rating ~ inference * model + (1|states) + (1|PID)) # not including random slope by states each state only has inference at two levels
stdCoef.merMod(mdl.hvm)
# visualize
# shows that the interaction was probably driven by tuncation in range
viz.hvm <- ggplot(dfhvm,aes(x = inference, y = rating)) +
  geom_point(aes(color = model)) +
  geom_smooth(aes(color = model),method = 'lm')


### visualize base-base correlation!
ggplot(
  data = dfhvm %>% 
          group_by(cond, states, model) %>% 
          dplyr::summarise(rating = mean(rating),inference = mean(inference)) %>% filter(cond == 'base'),
  aes(x = inference, y = rating,color = model)
) + 
  geom_point(aes(color = model)) + geom_smooth(method = 'lm')


