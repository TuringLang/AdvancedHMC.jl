### Mutable states

mutable struct NutpieHMCAdaptorState
    i::Int
    n_adapts::Int
    # The number of draws in the the early window
    early_end::Int
    # The first draw number for the final step size adaptation window
    final_step_size_window::Int

    function NutpieHMCAdaptorState(i, n_adapts, early_end, final_step_size_window)
        @assert (early_end < n_adapts) "Early_end must be less than num_tune (provided $early_end and $n_adapts)"
        return new(i, n_adapts, early_end, final_step_size_window)
    end
end
function NutpieHMCAdaptorState()
    return NutpieHMCAdaptorState(0, 1000, 300, 800)
end

function initialize!(state::NutpieHMCAdaptorState, early_window_share::Float64,
    final_step_size_window_share::Float64,
    n_adapts::Int)

    early_end = ceil(UInt64, early_window_share * n_adapts)
    step_size_window = ceil(UInt64, final_step_size_window_share * n_adapts)
    final_step_size_window = max(n_adapts - step_size_window, 0) + 1

    state.early_end = early_end
    state.n_adapts = n_adapts
    state.final_step_size_window = final_step_size_window
end

# function Base.show(io::IO, state::NutpieHMCAdaptorState)
#     print(io, "window($(state.window_start), $(state.window_end)), window_splits(" * string(join(state.window_splits, ", ")) * ")")
# end

### Nutpie's adaptation
# Acknowledgement: ...
struct NutpieHMCAdaptor{M<:MassMatrixAdaptor,Tssa<:StepSizeAdaptor} <: AbstractAdaptor
    pc::M
    ssa::Tssa
    early_window_share::Float64
    final_step_size_window_share::Float64
    mass_matrix_switch_freq::Int
    early_mass_matrix_switch_freq::Int
    state::NutpieHMCAdaptorState
end
# Base.show(io::IO, a::NutpieHMCAdaptor) =
#     print(io, "NutpieHMCAdaptor(\n    pc=$(a.pc),\n    ssa=$(a.ssa),\n    init_buffer=$(a.init_buffer), term_buffer=$(a.term_buffer), window_size=$(a.window_size),\n    state=$(a.state)\n)")

function NutpieHMCAdaptor(
    pc::ExpWeightedWelfordVar,
    ssa::StepSizeAdaptor;
    early_window_share::Float64=0.3,
    final_step_size_window_share::Float64=0.2,
    mass_matrix_switch_freq::Int=60,
    early_mass_matrix_switch_freq::Int=10
)
    return NutpieHMCAdaptor(pc, ssa, early_window_share, final_step_size_window_share, mass_matrix_switch_freq, early_mass_matrix_switch_freq, NutpieHMCAdaptorState())
end

# !Q: Is mass_matrix a variance or an inverse of it? It should be inverse, but accumulators are directly variance?
# forward the method to the current draw
# it will then be forwarded to `var` property
getM⁻¹(ca::NutpieHMCAdaptor) = getM⁻¹(ca.pc.exp_variance_draw)
getϵ(ca::NutpieHMCAdaptor) = getϵ(ca.ssa)

function initialize!(adaptor::NutpieHMCAdaptor, n_adapts::Int, z::PhasePoint)
    initialize!(adaptor.state, adaptor.early_window_share, adaptor.final_step_size_window_share, n_adapts)
    # !Q: Shall we initialize from the gradient?
    # Nutpie initializes the variance estimate with reciprocal of the gradient
    # Like: Some((grad).abs().recip().clamp(LOWER_LIMIT, UPPER_LIMIT))
    adaptor.pc.exp_variance_draw.var = (1 ./ abs.(z.ℓπ.gradient)) |> x -> clamp.(x, LOWER_LIMIT, UPPER_LIMIT)
    return adaptor
end
finalize!(adaptor::NutpieHMCAdaptor) = finalize!(adaptor.ssa)

#
is_in_first_step_size_window(ad::NutpieHMCAdaptor) = ad.state.i <= ad.state.final_step_size_window
is_in_early_window(ad) = ad.state.i <= ad.state.early_end
switch_freq(ad::NutpieHMCAdaptor) = is_in_early_window(ad) ? ad.early_mass_matrix_switch_freq : ad.mass_matrix_switch_freq
#
# Changes vs Rust implementation
# - step_size_adapt is at the top
# - several checks are handled in sampler (finalize adaption, do not adapt during normal sampling)
function adapt!(
    ad::NutpieHMCAdaptor,
    θ::AbstractVecOrMat{<:AbstractFloat},
    α::AbstractScalarOrVec{<:AbstractFloat},
    z::PhasePoint
)
    ad.state.i += 1

    adapt!(ad.ssa, θ, α)

    # TODO: do we resize twice? also in update?
    # !Q: Why do we check resizing several times during iteration? (also in adapt!)
    resize!(ad.pc, θ, z.ℓπ.gradient) # Resize pre-conditioner if necessary.

    # determine whether to update mass matrix
    if is_in_first_step_size_window(ad)

        # Switch swaps the background (_bg) values for currect, and resets the background values 
        # Frequency of the switch depends on the phase
        background_count(ad.pc) >= switch_freq(ad) && switch!(ad.pc)

        # TODO: implement a skipper for bad draws
        # !Q: Why does it always update? (as per Nuts-rs/Nutpie)
        adapt!(ad.pc, θ, α, z.ℓπ.gradient, true)
    end
end
# Missing: collector checks on divergences or terminating at idx=0 // finite and None esimtaor
# adapt! estimator only if collected stuff is good (divergences)
# init for mass_matrix is grad.abs().recip.clamp(LOWER_LIMIT, UPPER_LIMIT) // init of ExpWindowDiagAdapt
# add checks taht exp_vairances are the same length?
