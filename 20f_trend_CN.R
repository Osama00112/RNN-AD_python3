# Load required packages
library(ggplot2)
library(reshape2)
library(readr)
library(dplyr)

# Read the data from CSV file
data <- read_csv("22f_CN_trend.csv")

# Reshape the data to long format for ggplot2
data_long <- melt(data, id.vars = "Feature Name", 
                  variable.name = "Month", 
                  value.name = "Attribution")

# Convert the Month column to numeric
data_long$Month <- as.numeric(data_long$Month)
# 
# # Manually assign trend categories and colors
# data_long <- data_long %>%
#   mutate(Trend = case_when(
#     `Feature Name` == "DX: MCI" ~ "falling",
#     `Feature Name` == "RAVLT_immediate" ~ "consistent",
#     TRUE ~ "rising"
#   )) %>%
#   mutate(Color = case_when(
#     Trend == "rising" & `Feature Name` == "ABETA" ~ "orange",  # lighter red
#     Trend == "rising" & `Feature Name` == "FAQ" ~ "orange",  # medium red
#     Trend == "rising" & `Feature Name` == "DX: CN" ~ "red",  # dark red
#     Trend == "rising" & `Feature Name` == "FDG" ~ "gold",  # dark red
#     Trend == "rising" & `Feature Name` == "ADAS11" ~ "gold",  # dark red
#     Trend == "falling" & `Feature Name` == "DX: MCI" ~ "green",  # dark green
#     Trend == "consistent" & `Feature Name` == "RAVLT_immediate" ~ "blue"  # dark blue
#   ))
# 
# # Define the order of the legend items
# legend_order <- c("red", "orange", "gold", "blue", "green")
# 
# # Plot the trends
# ggplot(data_long, aes(x = Month, y = Attribution, group = `Feature Name`, color = Color)) +
#   geom_line(linewidth = 1.2) +  # Use linewidth instead of size for lines
#   geom_point(size = 3) +        # Plot points at each time step
#   scale_color_manual(
#     values = c("blue" = "blue", "gold" = "gold", "orange" = "orange", "red" = "red", "green" = "green"),
#     breaks = legend_order,
#     labels = c("DX: CN", "FAQ, ABETA", "ADAS11, FDG", "RAVLT_immediate", "DX: MCI")
#   ) +
#   theme_minimal(base_size = 14) +  # Increase base font size for better readability
#   labs(
#     title = "IG Feature Attribution Trends Over Time",
#     x = "Time (Years)",
#     y = "Attribution Score",
#     color = "Trend Category"
#   ) +



# Manually assign trend categories and colors
data_long <- data_long %>%
  mutate(Trend = case_when(
    `Feature Name` == "DX: MCI" ~ "falling",
    `Feature Name` == "RAVLT_immediate" ~ "consistent",
    TRUE ~ "rising"
  )) %>%
  mutate(Color = case_when(
    Trend == "rising" & `Feature Name` == "ABETA" ~ "orange",  # lighter red
    Trend == "rising" & `Feature Name` == "FAQ" ~ "orange",  # medium red
    Trend == "rising" & `Feature Name` == "DX: CN" ~ "red",  # dark red
    Trend == "rising" & `Feature Name` == "FDG" ~ "gold",  # dark red
    Trend == "rising" & `Feature Name` == "ADAS11" ~ "gold",  # dark red
    Trend == "falling" & `Feature Name` == "DX: MCI" ~ "green",  # dark green
    Trend == "consistent" & `Feature Name` == "RAVLT_immediate" ~ "blue"  # dark blue
  ))

# Define the order of the legend items
legend_order <- c("red", "orange", "gold", "blue", "green")

# Plot the trends
ggplot(data_long, aes(x = Month, y = Attribution, group = `Feature Name`, color = Color)) +
  geom_line(linewidth = 1.2) +  # Use linewidth instead of size for lines
  geom_point(size = 3) +        # Plot points at each time step
  scale_color_manual(
    values = c("blue" = "blue", "gold" = "gold", "orange" = "orange", "red" = "red", "green" = "green"),
    breaks = legend_order,
    labels = c("DX: CN", "FAQ, ABETA", "ADAS11, FDG", "RAVLT_immediate", "DX: MCI")
  ) +
  theme_minimal(base_size = 14) +  # Increase base font size for better readability
  labs(
    title = "IG Feature Attribution Trends Over Time",
    x = "Time (Years)",
    y = "Attribution Score",
    color = "Trend Category"
  )


  theme(
    text = element_text(family = "sans", color = "black"),  # Use a more common font
    plot.title = element_text(size = 16, face = "bold"),  # Make title larger
    axis.title = element_text(size = 12, face = "bold"),  # Bold axis titles
    axis.text = element_text(size = 10),  # Larger axis text
    legend.title = element_text(size = 12, face = "bold"),  # Bold legend title
    legend.text = element_text(size = 10)  # Larger legend text
  ) +
  coord_cartesian(ylim = c(min(data_long$Attribution, na.rm = TRUE), max(data_long$Attribution, na.rm = TRUE))) +  # Adjust axis limits
  theme(
    panel.grid.major = element_line(color = "gray90"),  # Light grid lines for readability
    panel.grid.minor = element_blank(),  # Remove minor grid lines for less clutter
    panel.border = element_rect(color = "gray80", fill = NA)  # Add a border around the plot
  )
