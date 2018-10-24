library(survnet)
library(survival)
source("simulate_sinus.R")
context("survnet_cr")

# RNN data
dat <- simulate_sinus_data(n = 100, p = 2, sequence_length = 12)

test_that("Specification of cause-specific layers works", {
  nn <- survnet(y = dat$y, x = dat$x, breaks = breaks, epochs = 2, 
                units_causes = c(3, 4, 5, 6), verbose = 0)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  expect_length(grep("cause1", layer_names), 4)
  expect_length(grep("cause2", layer_names), 4)
})