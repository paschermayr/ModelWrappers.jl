############################################################################################
"""
$(TYPEDEF)
Abstract super type for Supported Automatic Differentiation backends.
"""
abstract type AutomaticDifferentiationMethod end
struct ADForward <: AutomaticDifferentiationMethod end
struct ADZygote <: AutomaticDifferentiationMethod end
struct ADReverse <: AutomaticDifferentiationMethod end
struct ADReverseUntaped <: AutomaticDifferentiationMethod end

############################################################################################
#Helper function for AD packages
function checkfinite(θ::ReverseDiff.TrackedArray{T}, max_val::R=max_val) where {T,R<:Real}
    @inbounds @simd for iter in eachindex(θ)
        if !checkfinite(θ[iter], max_val=max_val)
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)
Initiate DiffResults.MutableDiffResult struct. Not exported.

# Examples
```julia
```

"""
function _diffresults_buffer(θᵤ::AbstractVector{T}) where {T<:Real}
    #NOTE: Adjusted from: https://github.com/tpapp/LogDensityProblems.jl/blob/master/src/DiffResults_helpers.jl
    S = T <: Real ? T : Float64
    return DiffResults.MutableDiffResult(zero(S), (Vector{S}(undef, size(θᵤ, 1)),))
end
############################################################################################
struct AutomaticDiffTune{M<:AutomaticDifferentiationMethod,C} <: AbstractDifferentiableTune
    "Automatic Differentiation (AD) backend."
    backend::M
    "Chunck size configuration for AD backend."
    config::C
    function AutomaticDiffTune(backend::M, config::C
    ) where {M<:AutomaticDifferentiationMethod,C}
        return new{M,C}(backend, config)
    end
end
function AutomaticDiffTune(backend::M, objective::Objective) where {M<:AutomaticDifferentiationMethod}
    return AutomaticDiffTune(
    backend, _config(backend, objective, unconstrain_flatten(objective.model, objective.tagged))
    )
end

############################################################################################
function update(
    tune::AutomaticDiffTune,
    objective::Objective)
    return AutomaticDiffTune(tune.backend, objective)
end

############################################################################################
function _log_density(
    objective::Objective, tune::AutomaticDiffTune, θᵤ::AbstractVector{T}
) where {T<:Real}
    return objective(θᵤ)
end

############################################################################################
#Make Symbol intialization easier
function AutomaticDiffTune(backend::Symbol, objective::Objective)
    return AutomaticDiffTune(Val(backend), objective)
end

############################################################################################
#!NOTE: All those functions need to be dispatched if new AD framework is included
"""
$(SIGNATURES)
Write config file for AD wrapper.

# Examples
```julia
```

"""
function _config(
    differentiation::ADForward, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return ForwardDiff.GradientConfig(objective, θᵤ)
end
function _config(
    differentiation::ADZygote, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return nothing
end
function _config(
    differentiation::ADReverse, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return ReverseDiff.compile(ReverseDiff.GradientTape(objective, θᵤ))
end
function _config(
    differentiation::ADReverseUntaped, objective::Objective, θᵤ::AbstractVector{R}
) where {R<:Real}
    return nothing
end

############################################################################################
function AutomaticDiffTune(
    backend::Val{:ForwardDiff},
    objective::Objective,
    config::C=_config(ADForward(), objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADForward(), config)
end

function AutomaticDiffTune(
    backend::Val{:Zygote},
    objective::Objective,
    config::C=_config(ADZygote(), objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADZygote(), config)
end

function AutomaticDiffTune(
    backend::Val{:ReverseDiff},
    objective::Objective,
    config::C=_config(ADReverse(), objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADReverse(), config)
end

function AutomaticDiffTune(
    backend::Val{:ReverseDiffUntaped},
    objective::Objective,
    config::C=_config(ADReverseUntaped(), objective, unconstrain_flatten(objective.model, objective.tagged)),
) where {C}
    return AutomaticDiffTune(ADReverseUntaped(), config)
end

############################################################################################
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADForward}, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_buffer(θᵤ)
    result = ForwardDiff.gradient!(buffer, objective, θᵤ, tune.config)
    return DiffResults.value(result), DiffResults.gradient(result)
end
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADZygote}, θᵤ::AbstractVector{T}
) where {T<:Real}
    _val, back = Zygote.pullback(objective, θᵤ)
    return T(_val), first(back(Zygote.sensitivity(_val)))
end
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADReverse}, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_buffer(θᵤ)
    result = ReverseDiff.gradient!(buffer, tune.config, θᵤ)
    return DiffResults.value(result), DiffResults.gradient(result)
end
function _log_density_and_gradient(
    objective::Objective, tune::AutomaticDiffTune{ADReverseUntaped}, θᵤ::AbstractVector{T}
) where {T<:Real}
    buffer = _diffresults_buffer(θᵤ)
    result = ReverseDiff.gradient!(buffer, objective, θᵤ)
    return DiffResults.value(result), DiffResults.gradient(result)
end

############################################################################################
# Export
export AutomaticDiffTune
