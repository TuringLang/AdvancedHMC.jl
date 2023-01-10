using StanHMCAdaptor

mutable struct DiagAdaptExpSettings <: StanHMCAdaptorSettings
    store_mass_matrix::Bool
end

DiagAdaptExpSettings() = DiagAdaptExpSettings(false)

mutable struct ExpWindowDiagAdapt{F} <: StanHMCAdaptor{F}
    dim::Int
    exp_variance_draw::RunningVariance
    exp_variance_grad::RunningVariance
    exp_variance_grad_bg::RunningVariance
    exp_variance_draw_bg::RunningVariance
    settings::DiagAdaptExpSettings
    _phantom::Phantom{F}
end

function ExpWindowDiagAdapt(dim::Int, settings::DiagAdaptExpSettings)
    ExpWindowDiagAdapt(dim, RunningVariance(dim), RunningVariance(dim), RunningVariance(dim), RunningVariance(dim), settings, Phantom{F}())
end

function update!(adaptor::ExpWindowDiagAdapt, state::StanHMCAdaptorState, collector::DrawGradCollector)
    if collector.is_good
        for i in 1:adaptor.dim
            adaptor.exp_variance_draw.add_sample(collector.draw[i])
            adaptor.exp_variance_grad.add_sample(collector.grad[i])
            adaptor.exp_variance_draw_bg.add_sample(collector.draw[i])
            adaptor.exp_variance_grad_bg.add_sample(collector.grad[i])
        end
    end
    if adaptor.exp_variance_draw.count() >= 3
        for i in 1:adaptor.dim
            diag = (adaptor.exp_variance_draw.current()[i] / adaptor.exp_variance_grad.current()[i])^0.5
            diag = max(LOWER_LIMIT, min(UPPER_LIMIT, diag))
            if isfinite(diag)
                state.mass_matrix.update_diag[i] = diag
            end
        end
    end
end

function initialize_state(adaptor::ExpWindowDiagAdapt)
    return StanHMCAdaptorState()
end

mutable struct ExpWindowDiagAdaptState <: StanHMCAdaptorState
    mass_matrix_inv::Union{Nothing,Vector{Float64}}
end

function create_adaptor_state(adaptor::ExpWindowDiagAdapt)
    return ExpWindowDiagAdaptState(nothing)
end

function sample_stats(adaptor::ExpWindowDiagAdapt, state::ExpWindowDiagAdaptState)
    ExpWindowDiagAdaptState(state.mass_matrix_inv)
end

# Grad

mutable struct GradDiagStrategy{F} <: StanHMCAdaptor{F}
    step_size::DualAverageStrategy{F,DiagMassMatrix}
    mass_matrix::ExpWindowDiagAdapt{F}
    options::GradDiagOptions
    num_tune::UInt64
    early_end::UInt64
    final_step_size_window::UInt64
end

mutable struct GradDiagOptions
    dual_average_options::DualAverageSettings
    mass_matrix_options::DiagAdaptExpSettings
    early_window::Float64
    step_size_window::Float64
    mass_matrix_switch_freq::UInt64
    early_mass_matrix_switch_freq::UInt64
end

GradDiagOptions() = GradDiagOptions(DualAverageSettings(), DiagAdaptExpSettings(), 0.3, 0.2, 60, 10)

mutable struct GradDiagStats <: StanHMCAdaptorStats
    step_size_stats::DualAverageStats
    mass_matrix_stats::ExpWindowDiagAdaptStats
end

function GradDiagStrategy(options::GradDiagOptions, num_tune::UInt64, dim::Int)
    num_tune_f = convert(Float64, num_tune)
    step_size_window = convert(UInt64, options.step_size_window
                                       *
                                       num_tune_f)
    early_end = convert(UInt64, options.early_window * num_tune_f)
    final_second_step_size = max(num_tune - convert(UInt64, step_size_window), 0)

    GradDiagStrategy(DualAverageStrategy(options.dual_average_options, num_tune, dim),
        ExpWindowDiagAdapt(dim, options.mass_matrix_options),
        options,
        num_tune,
        early_end,
        final_second_step_size)
end

function update!(adaptor::GradDiagStrategy, state::StanHMCAdaptorState, collector::DrawGradCollector)
    if collector.is_good
        step_size_stats = update!(adaptor.step_size, state, collector)
        mass_matrix_stats = update!(adaptor.mass_matrix, state, collector)
    end
    if adaptor.draw >= adaptor.num_tune
        return
    end
    if adaptor.draw < adaptor.final_step_size_window
        is_early = adaptor.draw < adaptor.early_end
        switch_freq = is_early ? adaptor.options.early_mass_matrix_switch_freq : adaptor.options.mass_matrix_switch_freq
        if adaptor.mass_matrix.background_count() >= switch_freq
            adaptor.mass_matrix.switch(collector)
        end
    end
    if adaptor.draw >= adaptor.final_step_size_window
        adaptor.mass_matrix.update_potential(potential)
    end
end

function initialize_state(adaptor::GradDiagStrategy)
    return initialize_state(adaptor.mass_matrix)
end

function create_adaptor_state(adaptor::GradDiagStrategy)
    return create_adaptor_state(adaptor.mass_matrix)
end

function sample_stats(adaptor::GradDiagStrategy, state::StanHMCAdaptorState)
    step_size_stats = sample_stats(adaptor.step_size, state)
    mass_matrix_stats = sample_stats(adaptor.mass_matrix, state)
    GradDiagStats(step_size_stats, mass_matrix_stats)
end