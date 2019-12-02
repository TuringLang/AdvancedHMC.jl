######################
### Mutable states ###
######################

mutable struct StanHMCAdaptorState
    i           :: Int
    in_window   :: Vector{Bool}
    window_end  :: Vector{Bool}
end

################
### Adaptors ###
################

# TODO: currently only StanHMCAdaptor has the filed `n_adapts`. maybe we could unify all
# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.
struct StanHMCAdaptor{M<:AbstractPreconditioner, Tssa<:StepSizeAdaptor} <: AbstractAdaptor
    pc          :: M
    ssa         :: Tssa
    init_buffer :: Int
    term_buffer :: Int
    window_size :: Int
    state       :: StanHMCAdaptorState
end
Base.show(io::IO, a::StanHMCAdaptor) =
    print(io, "StanHMCAdaptor(pc=$(a.pc), ssa=$(a.ssa), init_buffer=$(a.init_buffer), term_buffer=$(a.term_buffer))")

function StanHMCAdaptor(
    pc::M,
    ssa::StepSizeAdaptor,
    init_buffer::Int=75,
    term_buffer::Int=50,
    window_size::Int=25
) where {M<:AbstractPreconditioner}
    return StanHMCAdaptor(pc, ssa, init_buffer, term_buffer, window_size, StanHMCAdaptorState(0, Vector{Bool}(undef, 0), Vector{Bool}(undef, 0)))
end

# @enum WindowState winout=1 winin=2 winend=3
is_in_window(tp::StanHMCAdaptor) = tp.state.in_window[tp.state.i] 
is_window_end(tp::StanHMCAdaptor) = tp.state.window_end[tp.state.i]

getM⁻¹(adaptor::StanHMCAdaptor) = getM⁻¹(adaptor.pc)
getϵ(adaptor::StanHMCAdaptor)   = getϵ(adaptor.ssa)

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function init!(adaptor::StanHMCAdaptor, n_adapts::Int)
    init_buffer, term_buffer, window_size = adaptor.init_buffer, adaptor.term_buffer, adaptor.window_size

    # Compute in_window points
    in_window = Vector{Bool}(undef, n_adapts)
    in_window .= false
    in_window[init_buffer:n_adapts-term_buffer-1] .= true

    # Compute window_end points
    window_end = Vector{Bool}(undef, n_adapts)
    window_end .= false
    next_window = init_buffer + window_size - 1
    while next_window < n_adapts - term_buffer
        window_end[next_window] = true
        window_size *= 2
        next_window += window_size
        # Extend the current window to the end of the full window
        # if the next window reaches the end of the full window
        next_window_boundary = next_window + 2 * window_size
        if next_window_boundary >= n_adapts - term_buffer
            next_window = n_adapts - term_buffer - 1
        end
    end

    adaptor.state.i = 0
    adaptor.state.in_window = in_window
    adaptor.state.window_end = window_end
end
finalize!(adaptor::StanHMCAdaptor) = finalize!(adaptor.ssa)

function adapt!(
    tp::StanHMCAdaptor,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat}
)
    tp.state.i += 1

    adapt!(tp.ssa, θ, α)

    resize!(tp.pc, θ) # Resize pre-conditioner if necessary.

    # Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
    if is_in_window(tp)
        # We accumlate stats from θ online and only trigger the update of M⁻¹ in the end of window.
        is_update_M⁻¹ = is_window_end(tp)
        adapt!(tp.pc, θ, α, is_update_M⁻¹)
    end

    if is_window_end(tp)
        reset!(tp.ssa)
        reset!(tp.pc)
    end
end
