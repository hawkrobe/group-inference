require(shiny)
require(shinythemes)
require(ggplot2)
require(cowplot)

# source graphing functions and data loading
source('server_setup.R')

# define the UI component ---------------
ui <- fluidPage(theme = shinytheme('united'),
                tabsetPanel(
                  # group model compnents tab ------------
                  tabPanel('Group Model Components',
                           fluidRow(
                             column(4,
                                    h4('Group 1'),
                                    plotOutput('g.grp.comp.1',height = 500)
                             ),
                             column(4,
                                    h4('Group 2'),
                                    plotOutput('g.grp.comp.2',height = 500)    
                             ),
                             column(4,
                                    h4('Group 3'),
                                    plotOutput('g.grp.comp.3',height = 500)     
                             )
                           )
                  ),
                  # dimensional model componenets tab ----------
                  tabPanel('Dimension Model Components',
                           fluidRow(
                             column(4,
                                    h4('Dimension 1'),
                                    plotOutput('g.dim.comp.1',height = 500)
                             ),
                             column(4,
                                    h4('Dimension 2'),
                                    plotOutput('g.dim.comp.2',height = 500)    
                             ),
                             column(4,
                                    h4('Dimension 3'),
                                    plotOutput('g.dim.comp.3',height = 500)     
                             ) 
                           )
                  ),
                  # sparse-inference/mixture tab -------
                  tabPanel('Single Feature Sparse-Inference',
                          sidebarLayout(
                            # sidebar with all the input widgets
                            sidebarPanel(width = 3,
                              selectInput(inputId = 'fromstate',
                                          label = 'From',
                                          choices = unique(df$state1)
                              ),
                              uiOutput('tostate'), # conditional choiceset for to-state
                              sliderInput("tpinput", "Observed transition probability",
                                          min = 0.1, max = 0.9,value = 0.1,step = 0.1
                              )
                            ),
                            
                            mainPanel(
                              fluidRow(
                                column(4,
                                       h4('Hard Classification'),
                                       plotOutput('g.grp.h',height = 500)
                                ),
                                column(4,
                                       h4('Soft Classification'),
                                       plotOutput('g.grp.s',height = 500)
                                ),
                                column(4,
                                       h4('Dimension Mixture'),
                                       plotOutput('g.dim',height = 500)
                                )
                              )
                            )
                          )
                  )
          )
                
)

# define the server component -------------
server <- function(input, output){
  
  # rendering component heatmaps
  output$g.grp.comp.1 <- renderPlot(g.grp.comp.1)
  output$g.grp.comp.2 <- renderPlot(g.grp.comp.2)
  output$g.grp.comp.3 <- renderPlot(g.grp.comp.3)
  output$g.dim.comp.1 <- renderPlot(g.dim.comp.1)
  output$g.dim.comp.2 <- renderPlot(g.dim.comp.2)
  output$g.dim.comp.3 <- renderPlot(g.dim.comp.3)
  
  # conditional selectinput for to-state based on chosen fromstate
  output$tostate <- renderUI({
    set <- set_state$set[set_state$state1 == input$fromstate]
    ts <- set_state$state1[(set_state$set == set) & (set_state$state1 != input$fromstate)]
    selectInput('tostate',
                'To',
                choices = ts)
  })
  
  # hard group classification inference plot
  output$g.grp.h <- renderPlot(
    grp.h.heatmap(input$fromstate, input$tostate, input$tpinput)
  )
  
  # soft group classification inference
  output$g.grp.s <- renderPlot(
    grp.s.heatmap(input$fromstate, input$tostate, input$tpinput)
  )
  
  # dimension mixture inference
  output$g.dim <- renderPlot(
    dim.heatmap(input$fromstate, input$tostate, input$tpinput)
  )
}

shinyApp(ui, server)
