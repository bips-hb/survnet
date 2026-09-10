# survnet 0.0.7

* Switched from deprecated `keras` package to `keras3`.
* Use `py_require("tensorflow")` on package load to ensure TensorFlow is available.
* Updated loss function to use `keras3` ops (`op_sum`, `op_log`, `op_clip`) instead of legacy `backend()` API.
* Updated `optimizer_rmsprop()` to use `learning_rate` argument (replacing deprecated `lr`).
* Updated `regularizer_l2()` to use `l2` argument (replacing deprecated `l`).
* Removed `CUDNN_LSTM` and `CUDNN_GRU` RNN types (regular LSTM/GRU auto-use CuDNN in modern TensorFlow).
* Wrapped examples in `\donttest{}` and added test skips for environments without TensorFlow.
* Fixed DESCRIPTION metadata (Authors@R, title case, encoding, complete description).
* Used `\doi{}` instead of `\url{}` for DOI references in documentation.
