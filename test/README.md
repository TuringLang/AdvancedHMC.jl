* How to run tests locally

Assuming you are in the local folder of the root of AdvancedHMC.jl,
you can use the following command to run all tests locally:

``` sh
julia --project=@. -e 'using Pkg; Pkg.test(; test_args=ARGS)' 
```
If you are in a different folder, 
you can change `@.` to the root of AdvancedHMC.jl.

Further, the testing is set up to accept positional arguments to run a subset of tests by filtering.
For example, below runs only tests for `Adaptation`:

``` sh
julia --project=@. -e 'using Pkg; Pkg.test(; test_args=ARGS)' "Adaptation"
```

See [PR #287](https://github.com/TuringLang/AdvancedHMC.jl/pull/287) that introduces this functionality via [ReTest.jl](https://juliatesting.github.io/ReTest.jl/stable/) for more information.
