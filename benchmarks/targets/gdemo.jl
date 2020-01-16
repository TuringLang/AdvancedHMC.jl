# The most famous Turing model
# @model gdemo() = begin
#     s ~ InverseGamma(2, 3)
#     m ~ Normal(0, sqrt(s))
#     1.5 ~ Normal(m, sqrt(s))
#     2.0 ~ Normal(m, sqrt(s))
#     return s, m
# end

using Distributions: logpdf, InverseGamma, Normal
using Bijectors: invlink, logpdf_with_trans
using InteractiveUtils: @code_warntype

function get_gdemo()
    function invlink_gdemo(θ)
        s = invlink(InverseGamma(2, 3), θ[1])
        m = θ[2]
        return [s, m]
    end

    function ℓπ_gdemo(θ)
        s, m = invlink_gdemo(θ)
        logprior = logpdf_with_trans(InverseGamma(2, 3), s, true) + logpdf(Normal(0, sqrt(s)), m)
        loglikelihood = logpdf(Normal(m, sqrt(s)), 1.5) + logpdf(Normal(m, sqrt(s)), 2.0)
        return logprior + loglikelihood
    end

    return (ℓπ=ℓπ_gdemo, invlink=invlink_gdemo, θ̄=[49 / 24, 7 / 6])
end

function check_typed()
    ℓπ_gdemo, invlink_gdemo = get_gdemo()

    x = zeros(2)
    @code_warntype ℓπ_gdemo(x)
    @code_warntype invlink_gdemo(x)
end

if "--check_typed" in ARGS
    check_typed()
end
