
library(survnet)
library(survival)
source("simulate_sinus.R")
context("survnet_dropout")

# Test data
y <- Surv(veteran$time, veteran$status)
x <- veteran[, c(-2, -3, -4)]
breaks <- seq(0, 1000, length.out = 5)[-1]

# RNN data
dat <- simulate_sinus_data(n = 100, p = 2, sequence_length = 12)
breaks_rnn <- seq(0, 1, length.out = 5)[-1]

test_that("Dropout layer added", {
  nn <- survnet(y = y, x = x, breaks = breaks, epochs = 2, 
                units = c(3, 4), dropout = c(.4, 0), verbose = 0)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  cleaned_layer_names <- gsub("_.*$", "", layer_names)
  expect_equal(cleaned_layer_names, 
               c("input", "dense", "dropout", "dense", "output"))
})

test_that("Dropout layer added to RNN layers", {
  nn <- survnet(y = dat$y, x = dat$x, breaks = breaks_rnn, epochs = 2, 
                units_rnn = c(3, 4), dropout_rnn = c(.2, .3), verbose = 0, 
                skip = FALSE)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  cleaned_layer_names <- gsub("_.*$", "", layer_names)
  expect_equal(cleaned_layer_names, 
               c("input", "rnn", "dropout", "rnn", "dropout", 
                 "dense", "dense", "cause1", "cause2", "cause1", "cause2",
                 "concatenate", "output"))
})

test_that("No dropout layer added with defaults", {
  nn <- survnet(y = y, x = x, breaks = breaks, loss = loss_cif_loglik, epochs = 2, 
                verbose = 0)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  cleaned_layer_names <- gsub("_.*$", "", layer_names)
  expect_equal(cleaned_layer_names, 
               c("input", "dense", "dense", "output"))
})
