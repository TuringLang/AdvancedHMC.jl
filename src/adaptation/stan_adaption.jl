######################
### Mutable states ###
######################

mutable struct StanHMCAdaptorState
    i           :: Int
    window_size :: Int
    next_window :: Int
end

################
### Adaptors ###
################

# TODO: currently only StanHMCAdaptor has the filed `n_adapts`. maybe we could unify all
# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.
struct StanHMCAdaptor{M<:AbstractPreconditioner, Tssa<:StepSizeAdaptor} <: AbstractAdaptor
    n_adapts    :: Int
    pc          :: M
    ssa         :: Tssa
    init_buffer :: Int
    term_buffer :: Int
    state       :: StanHMCAdaptorState
end

function StanHMCAdaptor(
    n_adapts::Int,
    pc::M,
    ssa::StepSizeAdaptor,
    init_buffer::Int=75,
    term_buffer::Int=50,
    window_size::Int=25
) where {M<:AbstractPreconditioner}
    next_window = init_buffer + window_size - 1
    return StanHMCAdaptor(n_adapts, pc, ssa, init_buffer, term_buffer, StanHMCAdaptorState(0, window_size, next_window))
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function is_in_window(tp::StanHMCAdaptor)
    return (tp.state.i >= tp.init_buffer) &&
           (tp.state.i < tp.n_adapts - tp.term_buffer) &&
           (tp.state.i != tp.n_adapts)
end

function is_window_end(tp::StanHMCAdaptor)
    return (tp.state.i == tp.state.next_window) &&
           (tp.state.i != tp.n_adapts)
end

function compute_next_window!(tp::StanHMCAdaptor)
    if ~(tp.state.next_window == tp.n_adapts - tp.term_buffer - 1)
        tp.state.window_size *= 2
        tp.state.next_window = tp.state.i + tp.state.window_size
        if ~(tp.state.next_window == tp.n_adapts - tp.term_buffer - 1)
            next_window_boundary = tp.state.next_window + 2 * tp.state.window_size
            if (next_window_boundary >= tp.n_adapts - tp.term_buffer)
                tp.state.next_window = tp.n_adapts - tp.term_buffer - 1
            end
        end
    end
end

getM⁻¹(adaptor::StanHMCAdaptor) = getM⁻¹(adaptor.pc)
getϵ(adaptor::StanHMCAdaptor)   = getϵ(adaptor.ssa)
finalize!(adaptor::StanHMCAdaptor) = finalize!(adaptor.ssa)
# reset!(adaptor::StanHMCAdaptor) # Not used.

function adapt!(tp::StanHMCAdaptor, θ::AbstractVector{<:Real}, α::AbstractFloat)
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
        # If window ends, compute next window
        compute_next_window!(tp)
    end
end
