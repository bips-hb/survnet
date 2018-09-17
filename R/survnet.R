
#' Artificial neural networks for survival analysis
#'
#' @param y Survival outcome: \code{matrix}, \code{data.frame} or \code{Surv()} object.
#' @param x Predictors: \code{matrix}, \code{array} or \code{data.frame}.
#' @param breaks Right interval limits for discrete survival time.
#' @param units Vector of units, each specifying the number of units in one hidden layer.
#' @param units_rnn Vector of units for recurrent layers.
#' @param units_causes Vector of units for cause-specific layers (competing risks only). Either a vector (will be repeated for each cause) or a list of vectors with layers for each cause.
#' @param epochs Number of epochs to train the model.
#' @param batch_size Number of samples per gradient update.
#' @param validation_split Fraction in [0,1] of the training data to be used as validation data.
#' @param loss Loss function. 
#' @param activation Activtion function.
#' @param dropout Vector of dropout rates after each hidden layer. Use 0 for no dropout (default).
#' @param dropout_rnn Vector of dropout rates after each recurrent layer. Use 0 for no dropout (default).
#' @param dropout_causes Vector of dropout rates after each cause-specific layer. Use 0 for no dropout (default).
#' @param optimizer Name of optimizer or optimizer instance.
#' @param verbose Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch). 
#' 
#' @return Fitted model.
#' @export
#' @import survival keras
#' @importFrom magrittr freduce
survnet <- function(y, 
                    x, 
                    breaks, 
                    units = c(3, 5),
                    units_rnn = c(4, 6),
                    units_causes = c(3, 2),
                    epochs = 100, 
                    batch_size = 16,
                    validation_split = 0.2,
                    loss, 
                    activation = "tanh",
                    dropout = rep(0, length(units)),
                    dropout_rnn = rep(0, length(units_rnn)), 
                    dropout_causes = rep(0, length(units_causes)),
                    optimizer = optimizer_rmsprop(lr = 0.001), 
                    verbose = 2) {
  
  # TODO: Formula possible? Not for RNN?
  # model_data <- model.frame(formula, data)
  # y <- model_data[, 1]
  # x <- as.matrix(model_data[, -1])
  
  if (!(is.Surv(y) | ((is.numeric(y) | is.matrix(y) | is.data.frame(y)) & ncol(y) == 2))) {
    stop("Unexpected type for 'y'.")
  }
  
  if (!(is.matrix(x) | is.array(x))) {
    x <- as.matrix(x)
  }
  
  # Number of time intervals
  num_intervals <- length(breaks)
  
  # Competing risks?
  num_causes <- max(y[, 2])
  if (num_causes < 1) {
    stop("No events found.")
  }
  
  if (!is.list(units_causes)) {
    if (is.numeric(units_causes)) {
      units_causes <- rep(list(units_causes), num_causes)
    } else {
      stop("Misspecified cause-specific layers.")
    }
  }
  
  if (length(units_causes) != num_causes) {
    stop("Number of cause-specific layers not matching number of causes.")
  }

  # TODO: Check dropout specification
  # TODO: Allow list for cause-specific dropout
  
  # Convert data, depending on loss function
  if (identical(loss, loss_cif_loglik)) {
    y_mat <- convert_surv_cens(time = y[, 1], status = y[, 2], breaks = breaks, num_causes = num_causes)
  } else {
    stop("Unknown loss function.")
  }

  # Create model
  input <- layer_input(shape = dim(x)[-1])

  if (length(dim(x)) > 2) {
    # RNN layers
    rnn_layers <- lapply(1:length(units_rnn), function(i) {
      if (i == length(units_rnn)) {
        return_sequences <- FALSE
      } else {
        return_sequences <- TRUE
      }
      layer_lstm(units = units_rnn[i], activation = activation, return_sequences = return_sequences, 
                 name = paste0("rnn_", i))
    })
    # RNN Dropout layers
    for (i in 1:length(dropout_rnn)) {
      if (dropout_rnn[i] > 0) {
        rnn_layers <- append(rnn_layers, layer_dropout(rate = dropout_rnn[i]), i + length(rnn_layers) - length(units_rnn))
      }
    }
    # non-RNN layers
    dense_layers <- lapply(1:length(units), function(i) {
      layer_dense(units = units[i], activation = activation, name = paste0("dense_", i))
    })
    # non-RNN Dropout layers
    for (i in 1:length(dropout)) {
      if (dropout[i] > 0) {
        dense_layers <- append(dense_layers, layer_dropout(rate = dropout[i]), i + length(dense_layers) - length(units))
      }
    }
    shared <- magrittr::freduce(magrittr::freduce(input, rnn_layers), dense_layers)
  } else {
    # non-RNN layers
    layers <- lapply(1:length(units), function(i) {
      layer_dense(units = units[i], activation = activation, name = paste0("dense_", i))
    })
    # Dropout layers
    for (i in 1:length(dropout)) {
      if (dropout[i] > 0) {
        layers <- append(layers, layer_dropout(rate = dropout[i]), i + length(layers) - length(units))
      }
    }
    shared <- magrittr::freduce(input, layers)
  }

  if (num_causes> 1) {
    # Competing risk
    sub_layers <- lapply(1:num_causes, function(i) {
      # Cause-specific layers
      layers <- lapply(1:length(units_causes[[i]]), function(j) {
        layer_dense(units = units_causes[[i]][j], activation = activation, name = paste0("cause", i, "_", j))
      })
      # Dropout layers
      for (j in 1:length(dropout_causes)) {
        if (dropout_causes[j] > 0) {
          layers <- append(layers, layer_dropout(rate = dropout_causes[j]), j + length(layers) - length(units_causes[[i]]))
        }
      }
      magrittr::freduce(shared, layers)
    })
    output <- layer_concatenate(sub_layers) %>%
      layer_dense(units = num_causes * num_intervals, activation = 'softmax', name = "output")
  } else {
    # No competing risks
    output <- shared %>%
      layer_dense(units = num_intervals, activation = 'softmax', name = "output")
  }

  model <- keras_model(inputs = input, outputs = output)
  
  # Compile model
  model %>% compile(
    loss = loss_cif_loglik(num_intervals, num_causes),
    optimizer = optimizer
  )

  # Fit model
  history <- model %>% fit(
    x, y_mat,
    epochs = epochs, batch_size = batch_size, validation_split = validation_split, 
    verbose = verbose
  )

  # Return model
  res <- list(model = model, 
              history = history, 
              num_intervals = num_intervals, 
              num_causes = num_causes, 
              breaks = breaks)
  class(res) <- "survnet"
  res
}
