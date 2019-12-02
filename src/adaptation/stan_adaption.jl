######################
### Mutable states ###
######################

@enum WindowState winout=1 winin=2 winend=3

is_in_window(ws::WindowState) = ws != winout
is_window_end(ws::WindowState) = ws == winend

mutable struct StanHMCAdaptorState
    window  ::  Vector{WindowState}
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function init!(state::StanHMCAdaptorState, init_buffer::Int, term_buffer::Int, window_size::Int, n_adapts::Int)
    # Init by all out-window points  
    window = Vector{WindowState}(undef, n_adapts)
    window .= winout

    # Update in-window points
    window[init_buffer+1:end-term_buffer] .= winin

    # Update window-end points
    next_window = init_buffer + window_size
    while next_window <= n_adapts - term_buffer
        window[next_window] = winend
        window_size *= 2
        next_window += window_size
        # Extend the current window to the end of the full window
        # if the next window reaches the end of the full window
        next_window_boundary = next_window + 2 * window_size
        if next_window_boundary > n_adapts - term_buffer
            next_window = n_adapts - term_buffer
        end
    end

    # Make sure the last point is not end
    if window[end] == winend
        window[end] = winin
    end

    state.window = window
end

function get_win_starts_ends(window)
    winstarts, winends = [], []
    for i in 1:length(window)-1
        if window[i] != winin && is_in_window(window[i+1])
            push!(winstarts, i)
        end
        if is_window_end(window[i])
            push!(winends, i)
        end
    end
    return winstarts, winends
end

function Base.show(io::IO, state::StanHMCAdaptorState)
    window = state.window
    windows = length(window) > 0 ? (
        findfirst(ws -> ws == winin, window) - 1, 
        findall(ws -> ws == winend, state.window)...,
        (state.window[end] == winin ? tuple(length(state.window)) : tuple())...
    ) : tuple()
    print(io, "windows(" * string(join(windows, ", ")) * ")")
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
    print(io, "StanHMCAdaptor(\n    pc=$(a.pc),\n    ssa=$(a.ssa),\n    init_buffer=$(a.init_buffer),\n    term_buffer=$(a.term_buffer),\n    window_size=$(a.window_size),\n    state=$(a.state)\n)")


function StanHMCAdaptor(
    pc::M,
    ssa::StepSizeAdaptor;
    init_buffer::Int=75,
    term_buffer::Int=50,
    window_size::Int=25
) where {M<:AbstractPreconditioner}
    return StanHMCAdaptor(pc, ssa, init_buffer, term_buffer, window_size, StanHMCAdaptorState(Vector{WindowState}(undef, 0)))
end

getM⁻¹(adaptor::StanHMCAdaptor) = getM⁻¹(adaptor.pc)
getϵ(adaptor::StanHMCAdaptor)   = getϵ(adaptor.ssa)
function init!(adaptor::StanHMCAdaptor, n_adapts::Int)
    init!(adaptor.state, adaptor.init_buffer, adaptor.term_buffer, adaptor.window_size, n_adapts)
end
finalize!(adaptor::StanHMCAdaptor) = finalize!(adaptor.ssa)

function adapt!(
    tp::StanHMCAdaptor,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat}
)
    ws = popfirst!(tp.state.window)

    adapt!(tp.ssa, θ, α)

    resize!(tp.pc, θ) # Resize pre-conditioner if necessary.

    # Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
    if is_in_window(ws)
        # We accumlate stats from θ online and only trigger the update of M⁻¹ in the end of window.
        is_update_M⁻¹ = is_window_end(ws)
        adapt!(tp.pc, θ, α, is_update_M⁻¹)
    end

    if is_window_end(ws)
        reset!(tp.ssa)
        reset!(tp.pc)
    end
end
