using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n_dims"
            arg_type = Int
            required = true
        "--n_samples"
            arg_type = Int
            required = true
        "--n_chains"
            arg_type = Int
            required = true
        "--target"
            arg_type = String
            required = true
            range_tester = (x -> x in ("mog", "gauss"))
        "--use_xla"
            arg_type = Bool
            default = true
    end

    parsed_args = parse_args(s; as_symbols=true)

end