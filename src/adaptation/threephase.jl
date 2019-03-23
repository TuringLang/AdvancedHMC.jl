######################
### Mutable states ###
######################

mutable struct ThreePhaseState
    n           :: Int
    window_size :: Int
    next_window :: Int
end

################
### Adapterers ###
################

# TODO: currently only ThreePhaseAdapter has the filed `n_adapts`. maybe we could unify all
# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.
struct ThreePhaseAdapter <: AbstractCompositeAdapter
    n_adapts    :: Int
    pc          :: AbstractPreConditioner
    ssa         :: StepSizeAdapter
    init_buffer :: Int
    term_buffer :: Int
    state       :: ThreePhaseState
end

function ThreePhaseAdapter(n_adapts::Int, pc::AbstractPreConditioner, ssa::StepSizeAdapter,
                           init_buffer::Int=75, term_buffer::Int=50, window_size::Int=25)
    next_window = init_buffer + window_size - 1
    return ThreePhaseAdapter(n_adapts, pc, ssa, init_buffer, term_buffer, ThreePhaseState(0, window_size, next_window))
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function in_adaptation(tp::ThreePhaseAdapter)
    return (tp.state.n >= tp.init_buffer) &&
           (tp.state.n < tp.n_adapts - tp.term_buffer) &&
           (tp.state.n != tp.n_adapts)
end

function is_windowend(tp::ThreePhaseAdapter)
    return (tp.state.n == tp.state.next_window) &&
           (tp.state.n != tp.n_adapts)
end

function compute_next_window!(tp::ThreePhaseAdapter)
    if ~(tp.state.next_window == tp.n_adapts - tp.term_buffer - 1)
        tp.state.window_size *= 2
        tp.state.next_window = tp.state.n + tp.state.window_size
        if ~(tp.state.next_window == tp.n_adapts - tp.term_buffer - 1)
            next_window_boundary = tp.state.next_window + 2 * tp.state.window_size
            if (next_window_boundary >= tp.n_adapts - tp.term_buffer)
                tp.state.next_window = tp.n_adapts - tp.term_buffer - 1
            end
        end
    end
end

function adapt!(tp::ThreePhaseAdapter, θ::AbstractVector{<:Real}, α::AbstractFloat)
    if tp.state.n < tp.n_adapts
        tp.state.n += 1

        is_final = tp.state.n == tp.n_adapts
        adapt!(tp.ssa, θ, α, is_final)

        # Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
        # TODO: consider adding a filed in pc called `update_every`
        if in_adaptation(tp)
            is_update = is_windowend(tp)
            adapt!(tp.pc, θ, α, is_update)
        end

        if is_windowend(tp)
            reset!(tp.ssa)
            reset!(tp.pc)
        end

        # If window ends, compute next window
        is_windowend(tp) && compute_next_window!(tp)
    end
end
