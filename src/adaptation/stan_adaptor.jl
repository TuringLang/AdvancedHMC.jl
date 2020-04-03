### Mutable states

mutable struct StanHMCAdaptorState
    i               ::  Int
    window_start    ::  Int         # start of mass matrix adaptation
    window_end      ::  Int         #   end of mass matrix adaptation
    window_splits   ::  Vector{Int} # iterations to update metric using mass matrix
end

StanHMCAdaptorState() = StanHMCAdaptorState(0, 0, 0, Vector{Int}(undef, 0))

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function initialize!(state::StanHMCAdaptorState, init_buffer::Int, term_buffer::Int, window_size::Int, n_adapts::Int)
    window_start = init_buffer + 1
    window_end = n_adapts - term_buffer

    # Update window-end points
    window_splits = Vector{Int}(undef, 0)
    next_window = init_buffer + window_size
    while next_window <= window_end
        # Extend the current window to the end of the full window
        # if the next window reaches the end of the full window
        next_window_boundary = next_window + 2 * window_size
        if next_window_boundary > window_end
            next_window = window_end
        end
        # Include the window split
        push!(window_splits, next_window)
        # Expand window and compute the next split
        window_size *= 2
        next_window += window_size
    end
    # Avoid updating in the end
    if !isempty(window_splits)  # only when there is at least one split
        if window_splits[end] == n_adapts
            pop!(window_splits)
        end
    end

    state.window_start  = window_start
    state.window_end    = window_end
    state.window_splits = window_splits
end

function Base.show(io::IO, state::StanHMCAdaptorState)
    print(io, "window($(state.window_start), $(state.window_end)), window_splits(" * string(join(state.window_splits, ", ")) * ")")
end

### Stan's windowed adaptation

# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.
struct StanHMCAdaptor{M<:MassMatrixAdaptor, Tssa<:StepSizeAdaptor} <: AbstractAdaptor
    pc          :: M
    ssa         :: Tssa
    init_buffer :: Int
    term_buffer :: Int
    window_size :: Int
    state       :: StanHMCAdaptorState
end
Base.show(io::IO, a::StanHMCAdaptor) =
    print(io, "StanHMCAdaptor(\n    pc=$(a.pc),\n    ssa=$(a.ssa),\n    init_buffer=$(a.init_buffer), term_buffer=$(a.term_buffer), window_size=$(a.window_size),\n    state=$(a.state)\n)")

function StanHMCAdaptor(
    pc::MassMatrixAdaptor,
    ssa::StepSizeAdaptor;
    init_buffer::Int=75,
    term_buffer::Int=50,
    window_size::Int=25
)
    return StanHMCAdaptor(pc, ssa, init_buffer, term_buffer, window_size, StanHMCAdaptorState())
end

getM⁻¹(ca::StanHMCAdaptor) = getM⁻¹(ca.pc)
getϵ(ca::StanHMCAdaptor) = getϵ(ca.ssa)

function initialize!(adaptor::StanHMCAdaptor, n_adapts::Int)
    initialize!(adaptor.state, adaptor.init_buffer, adaptor.term_buffer, adaptor.window_size, n_adapts)
    return adaptor
end
finalize!(adaptor::StanHMCAdaptor) = finalize!(adaptor.ssa)

is_in_window(a::StanHMCAdaptor) = a.state.i >= a.state.window_start && a.state.i <= a.state.window_end
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
