# library(tidyverse)
# library(plotly)
# 
# data <- mtcars %>% mutate(cyl = factor(cyl),
#                           Model = rownames(mtcars))
# print(data)
# 
# 
# plot1 <- data %>% ggplot(aes(x = wt, y = mpg, size = hp)) + 
#                   geom_point(alpha = 0.5)
# plot1


# Load required packages
# library(ggplot2)
# library(readxl)
# 
# # Read the data from Excel file
# data <- read_excel("12.xlsx")
# 
# # Reshape the data to a long format
# data_long <- reshape2::melt(data, id.vars = "Feature Name", 
#                             measure.vars = c("CN Mean", "MCI Mean", "AD Mean"),
#                             variable.name = "Condition", 
#                             value.name = "Value")
# 
# # Create the bubble plot
# ggplot(data_long, aes(x = Condition, y = `Feature Name`, size = Value)) +
#   geom_point(alpha = 0.7) + 
#   theme_minimal() +
#   labs(title = "Bubble Plot of Feature Importance by Condition",
#        x = "Condition",
#        y = "Feature Name",
#        size = "Value") +
#   scale_size(range = c(1, 10)) # Adjust bubble size range if necessary
# 
library(ggplot2)
library(readxl)
library(reshape2)

# Read the data from Excel file
data <- read_excel("12.xlsx")

# Add columns for the color grouping for each condition
data$Color_CN <- c("red", "purple", "purple", "purple", "purple", "purple", 
                   "purple", "purple", "purple")
data$Color_MCI <- c("red", "purple", "green", "purple", "purple", "purple", 
                    "purple", "purple", "green")
data$Color_AD <- c("red", "purple", "red", "purple", "purple", "purple", 
                   "red", "green", "green")

# Reshape the data to a long format
data_long <- melt(data, id.vars = c("Feature Name", "Color_CN", "Color_MCI", "Color_AD"), 
                  measure.vars = c("CN", "MCI", "AD"),
                  variable.name = "Condition", 
                  value.name = "Value")

# Map the color for each condition based on the feature and condition
data_long$Color <- ifelse(data_long$Condition == "CN", data_long$Color_CN,
                          ifelse(data_long$Condition == "MCI", data_long$Color_MCI,
                                 data_long$Color_AD))

# Create a factor for ColorGroup with meaningful labels for the legend
data_long$ColorGroup <- factor(data_long$Color, 
                               levels = c("red", "purple", "green"), 
                               labels = c("Aligned with \nliterature", "Emerging new \npredictors", 
                                          "Relatively less \nsignificant"))

# Create the bubble plot
# ggplot(data_long, aes(x = Condition, y = `Feature Name`, size = Value, color = ColorGroup)) +
#   geom_point(alpha = 0.7) + 
#   theme_minimal() +
#   labs(title = "Bubble Plot of Feature Importance",
#        x = "Target Class",
#        y = "Feature Name",
#        size = "Attribution Score",
#        color = "Color Group") +  # Legend for color groups
#   scale_size(range = c(1, 13)) + # Adjust bubble size range if necessary
#   scale_color_manual(values = c("Group 1" = "red", "Group 2" = "purple")) + # Manually set colors
#   theme(
#     text = element_text(color = "black"), # Set font family and color
#     axis.title = element_text(size = 12), # Adjust size as needed
#     axis.text = element_text(size = 12), # Adjust size as needed
#     legend.title = element_text(size = 12), # Adjust size as needed
#     legend.text = element_text(size = 12) # Adjust size as needed
#   )
# 


ggplot(data_long, aes(x = `Feature Name`, y = Condition, size = Value, color = ColorGroup)) +
  geom_point(alpha = 0.7, stroke = 1.2) +  # Darken the bubble outlines and adjust thickness
  theme_minimal() +
  labs(title = "Bubble Plot of Feature Importance",
       x = "Feature Name",
       y = "Target Class",
       size = "Attribution Score",
       color = "Color Group") +  # Legend for color groups
  scale_size(range = c(1, 13)) + # Adjust bubble size range if necessary
  scale_color_manual(values = c("Aligned with \nliterature" = "red", "Emerging new \npredictors" = "purple", 
                                "Relatively less \nsignificant" = "green")) + # Manually set colors
  theme(
    text = element_text(color = "black"), # Set font family and color
    axis.title = element_text(size = 12), # Adjust size as needed
    axis.text.x = element_text(size = 9, angle = 90, hjust = 1), # Rotate x-axis labels
    axis.text.y = element_text(size = 9), # Adjust size as needed
    legend.title = element_text(size = 9), # Adjust size as needed
    legend.text = element_text(size = 9) # Adjust size as needed
  )





