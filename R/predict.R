
#' survnet prediction
#'
#' @param object \code{survnet} object
#' @param newdata New data predictors: \code{matrix}, \code{array} or \code{data.frame}.
#' @param cause Select cause for competing risks, \code{NULL} returns list of all causes.
#' @param ... Further arguments passed to or from other methods.
#'
#' @return Cumulative incidence function of selected or all causes.
#' @export
#' @importFrom stats predict
predict.survnet <- function(object, newdata, cause = NULL, ...) {
  if (is.data.frame(newdata)) {
    newdata <- as.matrix(newdata)
  }
  
  # Predict with Keras
  pred <- object$model %>% predict(newdata)
  
  # Cause-specific predictions
  cause_preds <- lapply(1:object$num_causes, function(i) {
    pred[, ((i - 1) * object$num_intervals + 1):(i * object$num_intervals)]
  })
  
  # Cause specific cumulative incidence functions, dim: Obs. X time
  cause_cifs <- lapply(cause_preds, function(x) {
    t(apply(x, 1, cumsum))
  })
  
  # Return all or selected CIFs
  if (object$num_causes == 1) {
    cause_cifs[[1]]
  } else if (is.null(cause)) {
    cause_cifs
  } else {
    cause_cifs[[cause]]
  }
}