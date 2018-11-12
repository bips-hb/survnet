library(survnet)
library(survival)
source("simulate_sinus.R")
context("survnet_cr")

# RNN data
dat <- simulate_sinus_data(n = 100, p = 2, sequence_length = 12)
breaks <- seq(0, 1, length.out = 5)[-1]

# Add non-RNN data
dat$xmean <- apply(dat$x, c(1, 3), mean)

test_that("Specification of cause-specific layers works", {
  nn <- survnet(y = dat$y, x = dat$x, breaks = breaks, epochs = 2, 
                units_causes = c(3, 4, 5, 6), verbose = 0)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  expect_length(grep("cause1", layer_names), 4)
  expect_length(grep("cause2", layer_names), 4)
})

test_that("Works with single cause-specific layer", {
  expect_silent(survnet(y = dat$y, x = dat$x, breaks = breaks, epochs = 2, 
                        units_causes = c(3), verbose = 0))
})

test_that("Combination of non-RNN and RNN data works", {
  # 4 concat layers with skip connections (1x input+RNN for shared, 1x input+RNN+shared for causes, 1x output)
  nn <- survnet(y = dat$y, x = list(dat$xmean, dat$x), breaks = breaks, epochs = 2, 
                units_causes = c(2, 3), verbose = 0)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  expect_length(grep("concatenate", layer_names), 3)
  
  # 2 concat layers without skip connections (1x input+RNN, 1x output)
  nn <- survnet(y = dat$y, x = list(dat$xmean, dat$x), breaks = breaks, epochs = 2, 
                units_causes = c(2, 3), verbose = 0, skip = FALSE)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  expect_length(grep("concatenate", layer_names), 2)
})

