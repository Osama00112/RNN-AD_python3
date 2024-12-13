# Load necessary library
library(ggplot2)

# Define legend labels and colors
legend_labels <- c("Rising", "Consistent", "Falling")
colors <- list(
  rising = colorRampPalette(c("red", "gold"))(10),
  consistent = "#0000FF",  # Blue color
  falling = colorRampPalette(c("darkgreen", "yellowgreen"))(10)
)

# Create a function to generate gradient color bars
gradient_bar <- function(colors, height = 1, width = 2) {
  gradient <- as.data.frame(seq_along(colors))
  ggplot(gradient, aes(x = seq_along(colors), y = 1, fill = as.factor(seq_along(colors)))) +
    geom_tile(width = width, height = height) +
    scale_fill_manual(values = colors) +
    theme_void() +
    theme(
      legend.position = "none",
      panel.grid = element_blank()
    )
}

# Create individual gradient bars for each legend entry
p_rising <- gradient_bar(colors$rising)
p_consistent <- gradient_bar(rep(colors$consistent, 10))
p_falling <- gradient_bar(colors$falling)

# Combine the plots into a legend
library(gridExtra)
grid.arrange(
  arrangeGrob(p_rising, p_consistent, p_falling, ncol = 1),
  top = textGrob("Legend", gp = gpar(fontsize = 16, fontface = "bold"))
)
