######################
### Mutable states ###
######################

mutable struct StanHMCAdaptorState
    i               ::  Int
    window_start    ::  Int
    window_end      ::  Int
    window_splits   ::  Vector{Int}
end

StanHMCAdaptorState() = StanHMCAdaptorState(0, 0, 0, Vector{Int}(undef, 0))

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function init!(state::StanHMCAdaptorState, init_buffer::Int, term_buffer::Int, window_size::Int, n_adapts::Int)
    state.window_start = init_buffer
    state.window_end = n_adapts - term_buffer
    # Update window-end points
    next_window = init_buffer + window_size
    while next_window <= n_adapts - term_buffer
        # Extend the current window to the end of the full window
        # if the next window reaches the end of the full window
        next_window_boundary = next_window + 2 * window_size
        if next_window_boundary > n_adapts - term_buffer
            next_window = n_adapts - term_buffer
        end
        # Include the window split
        push!(state.window_splits, next_window)
        # Expand window and compute the next split
        window_size *= 2
        next_window += window_size
    end
end

function Base.show(io::IO, state::StanHMCAdaptorState)
    print(io, "window($(state.window_start), $(state.window_end)), window_splits(" * string(join(state.window_splits, ", ")) * ")")
end

################
### Adaptors ###
################

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
    print(io, "StanHMCAdaptor(\n    pc=$(a.pc),\n    ssa=$(a.ssa),\n    init_buffer=$(a.init_buffer),\n    term_buffer=$(a.term_buffer),\n    window_size=$(a.window_size),\n    state=$(a.state)\n)")

function StanHMCAdaptor(
    pc::M,
    ssa::StepSizeAdaptor;
    init_buffer::Int=75,
    term_buffer::Int=50,
    window_size::Int=25
) where {M<:AbstractPreconditioner}
    return StanHMCAdaptor(pc, ssa, init_buffer, term_buffer, window_size, StanHMCAdaptorState())
end

getM⁻¹(adaptor::StanHMCAdaptor) = getM⁻¹(adaptor.pc)
getϵ(adaptor::StanHMCAdaptor)   = getϵ(adaptor.ssa)
function init!(adaptor::StanHMCAdaptor, n_adapts::Int)
    init!(adaptor.state, adaptor.init_buffer, adaptor.term_buffer, adaptor.window_size, n_adapts)
    return adaptor
end
finalize!(adaptor::StanHMCAdaptor) = finalize!(adaptor.ssa)

is_in_window(a::StanHMCAdaptor) = a.state.i > a.state.window_start && a.state.i <= a.state.window_end
is_window_end(a::StanHMCAdaptor) = a.state.i in a.state.window_splits

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

# Deprecated constructor
@deprecate StanHMCAdaptor(n_adapts, pc, ssa) init!(StanHMCAdaptor(pc, ssa), n_adapts)