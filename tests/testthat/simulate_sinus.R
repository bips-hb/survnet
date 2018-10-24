
# Simulate sinus data
simulate_sinus_data <- function(n, p, sequence_length, cens_prob = 0.3) {
  xseq <- seq(0, pi, length.out = sequence_length) + 100
  
  # Simulate random sinus waves
  freq <- replicate(p, runif(n, 0, 1))
  ampl <- replicate(p, runif(n, 0, 1))
  
  x <- sapply(1:p, function(i) {
    t(sin(xseq %*% t(pi * freq[, i]))) * ampl[, i]
  }, simplify = "array")
  
  # Cause 1: Risk based on frequency
  t1 <- exp(freq[, 1]) - 1
  
  # Cause 2: Risk based on amplitude
  t2 <- exp(ampl[, 1]) - 1
  
  # Time and event indicator
  tmin <- pmin(t1, t2)
  k <- 1*(t2 < t1) + 1
  
  # Random censoring 
  status <- rbinom(n, 1, 1-cens_prob)
  status[status > 0] <- k[status > 0]
  tcens <- sapply(tmin, runif, n = 1, min = 0)
  time <- tmin
  time[status == 0] <- tcens[status == 0]
  
  # Scale time to 0...1
  time <- time/max(time)
  time <- round(time, digits = 2)
  time <- pmin(time, 1-1e-7)
  
  # Final data. x: Array with dimensions n*time*p
  y_out <- as.matrix(data.frame(time = time, status = status))
  x_out <- array(x, dim = c(n, sequence_length, p))
  list(y = y_out, x = x_out, freq = freq, ampl = ampl)
}
