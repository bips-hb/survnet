
library(survnet)
library(survival)
context("survnet_dropout")

# Test data
y <- Surv(veteran$time, veteran$status)
x <- veteran[, c(-2, -3, -4)]
breaks <- seq(0, max(y[, 1]), length.out = 5)[-1]
breaks[5] <- Inf

test_that("Dropout layer added", {
  nn <- survnet(y = y, x = x, breaks = breaks, epochs = 2, 
                units = c(3, 4), dropout = c(.4, 0), verbose = 0)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  cleaned_layer_names <- gsub("_.*$", "", layer_names)
  expect_equal(cleaned_layer_names, 
               c("input", "dense", "dropout", "dense", "output"))
})

test_that("No dropout layer added with defaults", {
  nn <- survnet(y = y, x = x, breaks = breaks, loss = loss_cif_loglik, epochs = 2, 
                verbose = 0)
  layer_names <- sapply(nn$model$layers, function(x) x$name)
  cleaned_layer_names <- gsub("_.*$", "", layer_names)
  expect_equal(cleaned_layer_names, 
               c("input", "dense", "dense", "output"))
})
