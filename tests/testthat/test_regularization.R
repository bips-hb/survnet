
library(survnet)
library(survival)
source("simulate_sinus.R")
context("survnet_regularization")

# Test data
y <- Surv(veteran$time, veteran$status)
x <- veteran[, c(-2, -3, -4)]
breaks <- seq(0, max(y[, 1]), length.out = 5)[-1]
breaks[5] <- Inf

# RNN data
dat <- simulate_sinus_data(n = 100, p = 2, sequence_length = 12)

test_that("Regularization added", {
  nn <- survnet(y = y, x = x, breaks = breaks, epochs = 2, 
                units = c(3, 4, 4), l2 = c(.4, 0, .1), verbose = 0)
  expect_equal(as.numeric(nn$model$layers[[2]]$kernel_regularizer$l2), .4)
  expect_null(nn$model$layers[[3]]$kernel_regularizer)
  expect_equal(as.numeric(nn$model$layers[[4]]$kernel_regularizer$l2), .1)
})

test_that("Regularization added to RNN layers", {
  nn <- survnet(y = dat$y, x = dat$x, breaks = breaks, epochs = 2, 
                units_rnn = c(3, 4, 4), l2_rnn = c(.4, 0, .1), verbose = 0)
  expect_equal(as.numeric(nn$model$layers[[2]]$kernel_regularizer$l2), .4)
  expect_null(nn$model$layers[[3]]$kernel_regularizer)
  expect_equal(as.numeric(nn$model$layers[[4]]$kernel_regularizer$l2), .1)
})

test_that("No regularization added with defaults", {
  nn <- survnet(y = y, x = x, breaks = breaks, loss = loss_cif_loglik, epochs = 2, 
                verbose = 0)
  expect_null(nn$model$layers[[2]]$kernel_regularizer)
  expect_null(nn$model$layers[[3]]$kernel_regularizer)
})
