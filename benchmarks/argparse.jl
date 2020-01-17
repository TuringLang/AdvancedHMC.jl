using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n_dims"
            help = "Dimensionality of target space"
            arg_type = Int
            nargs = '*'
            required = true
        "--n_samples"
            help = "Number of samples to draw"
            arg_type = Int
            required = true
        "--n_chains"
            help = "Number of chains to run"
            arg_type = Int
            nargs = '*'
            required = true
        "--target"
            help = "Target distribution"
            arg_type = String
            required = true
            range_tester = (x -> x in ("mog", "gaussian"))
        "--ad"
            help = "AD backend to use (AdvancedHMC only)"
            arg_type = String
            default  = "zygote"
            range_tester = (x -> x in ("zygote", "forwarddiff", "reversediff", "hand"))
        "--n_runs"
            help = "Number of runs to perform"
            arg_type = Int
            default  = 3
        "--no_xla"
            help ="Disable XLA compilation for TensorFLow Probability"
            action = :store_false
        "--check_typed"
            help ="Whether or not to check type stability for Julia models"
            action = :store_true
    end

    parsed_args = parse_args(s; as_symbols=true)

end
