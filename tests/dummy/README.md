This directory is intended to hold a very small GGUF "dummy" model for CI smoke tests.

To produce a dummy artifact locally for testing, place a file named `ggml-model-i2_s.gguf` here and create a GitHub artifact named `dummy-model` in your CI run.

Note: actual model files are large and should not be committed to the repo.
