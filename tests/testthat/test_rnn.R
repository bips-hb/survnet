library(survnet)
library(survival)
source("simulate_sinus.R")
context("survnet_rnn")

# RNN data
dat <- simulate_sinus_data(n = 100, p = 2, sequence_length = 12)
breaks <- seq(0, 1, length.out = 5)[-1]

test_that("Selection of GRU works", {
  nn <- survnet(y = dat$y, x = dat$x, breaks = breaks, epochs = 2, 
                units_causes = c(3, 4, 5, 6), verbose = 0, 
                rnn_type = "GRU")
  expect_equal(as.character(nn$model$layers[[2]]), 
               "<keras.layers.recurrent.GRU>")
})
