library(survnet)
library(survival)
source("simulate_sinus.R")

# RNN data
dat <- simulate_sinus_data(n = 100, p = 2, sequence_length = 12)
breaks <- seq(0, 1, length.out = 5)[-1]

test_that("Selection of GRU works", {
  skip_if_no_tensorflow()
  nn <- survnet(y = dat$y, x = dat$x, breaks = breaks, epochs = 2,
                units_causes = c(3, 4, 5, 6), verbose = 0,
                rnn_type = "GRU")
  layer_class <- class(nn$model$layers[[2]])[1]
  expect_true(grepl("GRU", layer_class, ignore.case = TRUE))
})
