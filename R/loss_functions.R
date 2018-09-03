
#' Cumulative incidence log-likelihood
#' 
#' Likelihood of parametric inference for the cumulative incidence functions as defined by Jeong & Fine 2006. Also used by Lee et al. 2018.
#' 
#' Data structure: 
#' 
#' \code{y_true} True survival: Matrix with at-risk and event information. Format: (S_1, ..., S_K, E_1, ..., E_K). Dimensions: obs X 2*causes*time.
#' 
#' \code{y_pred} Network output: One probability for each time and cause. Format: (y_11, ..., y_T1, ..., y_TK). Dimensions: obs X causes*time.
#'
#' @param num_intervals Number of time intervals
#' @param num_causes Number of causes for competing risks
#'
#' @return Negative log-likelihood
#' @export
#' @references 
#'  \itemize{
#'   \item Jeong, J. & Fine, J. (2006). Direct parametric inference for the cumulative incidence function. J R Stat Soc Ser C Appl Stat 55:187-200. \url{https://doi.org/10.1111/j.1467-9876.2006.00532.x}.
#'   \item Lee, C., Zame, W.R., Yoon, J. & van der Shaar, M. (2018). DeepHit: A deep learning approach to survival analysis with competing risks. AAAI 2018. \url{http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit}.
#'  }
loss_cif_loglik <- function(num_intervals, num_causes = 1){
  function(y_true, y_pred) {
    K <- backend()
    
    # Survival and event indicators
    S <- y_true[, 1:(num_causes * num_intervals)] # Survival
    E <- y_true[, (num_causes * num_intervals + 1):(2 * num_causes * num_intervals)] # Events
    
    # Likelihood part for uncensored and censored observations (0 for censored)
    uncens <- K$sum(E * K$log(K$clip(y_pred, K$epsilon(), NULL)), axis = -1L)
    delta <- 1 - K$sum(E, axis = -1L)
    cens <- delta * K$log(K$clip(1 - K$sum(S * y_pred, axis = -1L), K$epsilon(), NULL))
    
    # Return negative log-likelihood 
    -(uncens + cens)
  }
}