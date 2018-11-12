
#' Create binary response matrix for survival data
#' 
#' Creates matrix with at-risk and event information. Format: (S_1, ..., S_K, E_1, ..., E_K). Dimensions: obs X 2*causes*time.
#'
#' @param time Survival time
#' @param status Censoring indicator: 0 for censored observations, positive values for events.
#' @param breaks Right interval limits for discrete survival time.
#' @param num_causes Number of competing risks.
#'
#' @return Binary response matrix.
#' @export
convert_surv_cens <- function(time, status, breaks, num_causes) {
  n <- length(time)
  S <- array(0, dim = c(n, length(breaks), num_causes))
  E <- array(0, dim = c(n, length(breaks), num_causes))
  warn <- FALSE
  for (i in 1:n) {
    idx <- time[i] > breaks
    S[i, which(idx), ] <- 1
    
    # Set S=0 in event interval
    if (any(!idx)) {
      S[i, min(which(!idx)), ] <- 0
      
      # Set event
      if (status[i] > 0) {
        E[i, min(which(!idx)), status[i]] <- 1
      } 
    } else {
      warn = TRUE
    }
  }
  
  if (warn) {
    warning("One or more event times larger than right-most interval limit, setting to censored.")
  }
  
  # Reshape
  S <- do.call(cbind, lapply(1:num_causes, function(i) S[,,i]))
  E <- do.call(cbind, lapply(1:num_causes, function(i) E[,,i]))
  cbind(S, E)
}
