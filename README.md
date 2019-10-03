# survnet: Artificial neural networks for survival analysis
Marvin N. Wright

## Installation
```R
devtools::install_github("bips-hb/survnet")
```

## Example
```R
library(survival)
library(survnet)

# Survival data
y <- veteran[, c(3, 4)]
x <- veteran[, c(-2, -3, -4)]
x <- data.frame(lapply(x, scale))
breaks <- c(1, 50, 100, 200, 500, 1000)

# Fit simple model
fit <- survnet(y = y, x = x, breaks = breaks)
plot(fit$history)
```