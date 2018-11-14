
#' Artificial neural networks for survival analysis
#'
#' @param y Survival outcome: \code{matrix}, \code{data.frame} or \code{Surv()} object.
#' @param x Predictors: \code{matrix}, \code{data.frame} or \code{array} (time-series). Also accepts a list of \code{matrix/data.frame} and \code{array} for both time-constant and time-series predictors.
#' @param breaks Right interval limits for discrete survival time.
#' @param units Vector of units, each specifying the number of units in one hidden layer.
#' @param units_rnn Vector of units for recurrent layers.
#' @param units_causes Vector of units for cause-specific layers (competing risks only). Either a vector (will be repeated for each cause) or a list of vectors with layers for each cause.
#' @param epochs Number of epochs to train the model.
#' @param batch_size Number of samples per gradient update.
#' @param validation_split Fraction in [0,1] of the training data to be used as validation data.
#' @param loss Loss function. 
#' @param activation Activation function.
#' @param rnn_type Type of RNN layers. Either \code{"LSTM"} (default), \code{"GRU"}, \code{"CUDNN_LSTM"} or \code{"CUDNN_GRU"}.
#' @param skip Add skip connection from input and RNN layers to cause-specific layers.
#' @param dropout Vector of dropout rates after each hidden layer. Use 0 for no dropout (default).
#' @param dropout_rnn Vector of dropout rates after each recurrent layer. Use 0 for no dropout (default).
#' @param dropout_causes Vector of dropout rates after each cause-specific layer. Use 0 for no dropout (default).
#' @param l2 Vector of L2 regularization factors for each hidden layer. Use 0 for no regularization (default).
#' @param l2_rnn Vector of L2 regularization factors for each recurrent layer. Use 0 for no regularization (default).
#' @param l2_causes Vector of L2 regularization factors for each cause-specific layer. Use 0 for no regularization (default).
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
                    loss = loss_cif_loglik, 
                    activation = "tanh",
                    rnn_type = "LSTM",
                    skip = TRUE,
                    dropout = rep(0, length(units)),
                    dropout_rnn = rep(0, length(units_rnn)), 
                    dropout_causes = rep(0, length(units_causes)),
                    l2 = rep(0, length(units)), 
                    l2_rnn = rep(0, length(units_rnn)), 
                    l2_causes = rep(0, length(units_causes)), 
                    optimizer = optimizer_rmsprop(lr = 0.001), 
                    verbose = 2) {
  
  # Force evaluation of dependent arguments
  force(dropout_causes)
  force(l2_causes)
  
  if (!(is.Surv(y) | ((is.numeric(y) | is.matrix(y) | is.data.frame(y)) & ncol(y) == 2))) {
    stop("Unexpected type for 'y'.")
  }
  
  # Unlist predictors
  if (is.list(x) && !is.data.frame(x)) {
    if (length(x) != 2 || !is.array(x[[2]])) {
      stop("Parameter 'x' must be matrix, data.frame, array or list of exactly one matrix/data.frame and one array.")
    } 
    x_rnn <- x[[2]]
    x <- x[[1]]
  } else if (is.data.frame(x) || is.matrix(x)) {
    x_rnn <- NULL
  } else if (is.array(x)) {
    x_rnn <- x
    x <- NULL
  }
  
  if (!is.null(x) && is.data.frame(x)) {
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
  
  if (!is.list(dropout_causes)) {
    if (is.numeric(dropout_causes)) {
      dropout_causes <- rep(list(dropout_causes), num_causes)
    } else {
      stop("Misspecified cause-specific dropout.")
    }
  }
  
  if (!is.list(l2_causes)) {
    if (is.numeric(l2_causes)) {
      l2_causes <- rep(list(l2_causes), num_causes)
    } else {
      stop("Misspecified cause-specific L2 regularization.")
    }
  }
  
  if (length(units_causes) != num_causes) {
    stop("Number of cause-specific layers not matching number of causes.")
  }
  
  if (!is.vector(dropout) || length(dropout) != length(units)) {
    stop("Number of dropout layers not matching number of layers.")
  }
  if (!is.vector(dropout_rnn) || length(dropout_rnn) != length(units_rnn)) {
    stop("Number of RNN dropout layers not matching number of RNN layers.")
  }
  if (!is.list(dropout_causes) || length(dropout_causes) != num_causes) {
    stop("Number of cause-specific dropout layers not matching number of causes.")
  } 
  if (!all(sapply(dropout_causes, length) == sapply(units_causes, length))) {
    stop("Number of cause-specific dropout layers not matching number of cause-specific layers.")
  }
  
  if (!is.vector(l2) || length(l2) != length(units)) {
    stop("Number of L2 regularization factors not matching number of layers.")
  }
  if (!is.vector(l2_rnn) || length(l2_rnn) != length(units_rnn)) {
    stop("Number of RNN L2 regularization factors not matching number of RNN layers.")
  }
  if (!is.list(l2_causes) || length(l2_causes) != num_causes) {
    stop("Number of cause-specific L2 regularization factors not matching number of causes.")
  }
  if (!all(sapply(l2_causes, length) == sapply(units_causes, length))) {
    stop("Number of cause-specific L2 regularization factors not matching number of cause-specific layers.")
  }
  
  # Convert data, depending on loss function
  if (identical(loss, loss_cif_loglik)) {
    y_mat <- convert_surv_cens(time = y[, 1], status = y[, 2], breaks = breaks, num_causes = num_causes)
  } else {
    stop("Unknown loss function.")
  }

  # Input layer
  if (!is.null(x)) {
    input <- layer_input(shape = dim(x)[-1])
  }

  # Define RNN layers
  if (!is.null(x_rnn)) {
    # Input layer
    input_rnn <- layer_input(shape = dim(x_rnn)[-1])
    # RNN layers
    rnn_layers <- lapply(1:length(units_rnn), function(i) {
      if (i == length(units_rnn)) {
        return_sequences <- FALSE
      } else {
        return_sequences <- TRUE
      }
      if (l2_rnn[i] > 0) {
        kernel_regularizer <- regularizer_l2(l = l2_rnn[i])
      } else {
        kernel_regularizer <- NULL
      }
      if (rnn_type == "LSTM") {
        layer_lstm(units = units_rnn[i], activation = activation, return_sequences = return_sequences, 
                   kernel_regularizer = kernel_regularizer, name = paste0("rnn_", i))
      } else if (rnn_type == "GRU") {
        layer_gru(units = units_rnn[i], activation = activation, return_sequences = return_sequences, 
                   kernel_regularizer = kernel_regularizer, name = paste0("rnn_", i))
      } else if (rnn_type == "CUDNN_LSTM") {
        layer_cudnn_lstm(units = units_rnn[i], return_sequences = return_sequences, 
                  kernel_regularizer = kernel_regularizer, name = paste0("rnn_", i))
      } else if (rnn_type == "CUDNN_GRU") {
        layer_cudnn_gru(units = units_rnn[i], return_sequences = return_sequences, 
                         kernel_regularizer = kernel_regularizer, name = paste0("rnn_", i))
      } else {
        stop("Unknown rnn_type.")
      }
      
    })
    # RNN Dropout layers
    for (i in 1:length(dropout_rnn)) {
      if (dropout_rnn[i] > 0) {
        rnn_layers <- append(rnn_layers, layer_dropout(rate = dropout_rnn[i]), i + length(rnn_layers) - length(units_rnn))
      }
    }
  } 
  
  # non-RNN layers
  dense_layers <- lapply(1:length(units), function(i) {
    if (l2[i] > 0) {
      kernel_regularizer <- regularizer_l2(l = l2[i])
    } else {
      kernel_regularizer <- NULL
    }
    layer_dense(units = units[i], activation = activation, 
                kernel_regularizer = kernel_regularizer, name = paste0("dense_", i))
  })
  # non-RNN Dropout layers
  for (i in 1:length(dropout)) {
    if (dropout[i] > 0) {
      dense_layers <- append(dense_layers, layer_dropout(rate = dropout[i]), i + length(dense_layers) - length(units))
    }
  }
  
  # Connect input
  if (!is.null(x) & is.null(x_rnn)) {
    # Only non-RNN
    shared <- magrittr::freduce(input, dense_layers)
  } else if (is.null(x) & !is.null(x_rnn)) {
    # Only RNN
    shared <- magrittr::freduce(magrittr::freduce(input_rnn, rnn_layers), dense_layers)
  } else {
    # Both non-RNN and RNN
    shared <-  magrittr::freduce(layer_concatenate(list(input, magrittr::freduce(input_rnn, rnn_layers))), 
                                 dense_layers)
  }

  if (num_causes > 1) {
    # Competing risk
    
    # Skip connections
    if (skip) {
      if (!is.null(x) & is.null(x_rnn)) {
        # Only non-RNN
        shared <- layer_concatenate(list(input, shared))
      } else if (is.null(x) & !is.null(x_rnn)) {
        # Only RNN
        shared <- layer_concatenate(list(magrittr::freduce(input_rnn, rnn_layers), shared))
      } else {
        # Both non-RNN and RNN
        shared <- layer_concatenate(list(input, magrittr::freduce(input_rnn, rnn_layers), shared))
      }
    }
    
    # Cause-specific layers
    sub_layers <- lapply(1:num_causes, function(i) {
      # Cause-specific layers
      layers <- lapply(1:length(units_causes[[i]]), function(j) {
        if (l2_causes[[i]][j] > 0) {
          kernel_regularizer <- regularizer_l2(l = l2_causes[[i]][j])
        } else {
          kernel_regularizer <- NULL
        }
        layer_dense(units = units_causes[[i]][j], activation = activation, 
                    kernel_regularizer = kernel_regularizer, name = paste0("cause", i, "_", j))
      })
      # Dropout layers
      for (j in 1:length(dropout_causes[[i]])) {
        if (dropout_causes[[i]][j] > 0) {
          layers <- append(layers, layer_dropout(rate = dropout_causes[[i]][j]), j + length(layers) - length(units_causes[[i]]))
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

  # Define model
  if (!is.null(x) & is.null(x_rnn)) {
    # Only non-RNN
    model <- keras_model(inputs = input, outputs = output)
    xx <- x
  } else if (is.null(x) & !is.null(x_rnn)) {
    # Only RNN
    model <- keras_model(inputs = input_rnn, outputs = output)
    xx <- x_rnn
  } else {
    # Both non-RNN and RNN
    model <- keras_model(inputs = list(input, input_rnn), outputs = output)
    xx <- list(x, x_rnn)
  }
  
  # Compile model
  model %>% compile(
    loss = loss_cif_loglik(num_intervals, num_causes),
    optimizer = optimizer
  )

  # Fit model
  history <- model %>% fit(
    xx, y_mat,
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
