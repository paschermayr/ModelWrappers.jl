############################################################################################
"""
$(TYPEDEF)
Abstract super type for AbstractDifferentiableObjective results.
"""
abstract type ℓObjectiveResult <: BaytesCore.AbstractResult end

"""
$(TYPEDEF)
Stores result for log density and parameter for 'ℓobjective' evaluation at 'parameter'.

# Fields
$(TYPEDFIELDS)
"""
struct ℓDensityResult{T,S} <: ℓObjectiveResult
    "Parameter in unconstrained space."
    θᵤ::T
    "Log density at θᵤ."
    ℓθᵤ::S
    function ℓDensityResult(θᵤ::AbstractVector{S}, ℓθᵤ::S) where {S<:Real}
        return new{typeof(θᵤ),S}(θᵤ, ℓθᵤ)
    end
end
function ℓDensityResult(objective::Objective, θᵤ::AbstractVector{T}) where {T<:Real}
    return ℓDensityResult(θᵤ, T(objective(θᵤ)))
end
function ℓDensityResult(objective::Objective)
    return ℓDensityResult(objective, unconstrain_flatten(objective.model, objective.tagged))
end
ℓDensityResult(diff::DiffObjective) = ℓDensityResult(diff.objective)

"""
$(TYPEDEF)
Stores result for log density, gradient, and parameter for 'ℓobjective' evaluation at 'parameter'.

# Fields
$(TYPEDFIELDS)
"""
struct ℓGradientResult{T,S,G} <: ℓObjectiveResult
    "Parameter in unconstrained space."
    θᵤ::T
    "Log density at θᵤ."
    ℓθᵤ::S
    "Gradient of log density at θᵤ."
    ∇ℓθᵤ::G
    function ℓGradientResult(
        θᵤ::T, ℓθᵤ::S, ∇ℓθᵤ::G
    ) where {T<:AbstractVector,S<:Real,G<:AbstractVector}
        @argcheck length(θᵤ) == length(∇ℓθᵤ)
        return new{T,S,G}(θᵤ, ℓθᵤ, ∇ℓθᵤ)
    end
end
function log_density(
    objective::Objective,
    θᵤ::AbstractVector{T}=unconstrain_flatten(objective.model, objective.tagged),
) where {T<:Real}
    ℓθᵤ = objective(θᵤ)
    if isfinite(ℓθᵤ)
        ℓDensityResult(θᵤ, T(ℓθᵤ))
    else
        ℓDensityResult(θᵤ, T(-Inf))
    end
end
function log_density_and_gradient(
    objective::Objective,
    tune::AbstractDifferentiableTune,
    θᵤ::AbstractVector{T}=unconstrain_flatten(objective.model, objective.tagged),
) where {T<:Real}
    ℓθᵤ, ∇ℓθᵤ = _log_density_and_gradient(objective, tune, θᵤ)
    if isfinite(ℓθᵤ)
        ℓGradientResult(θᵤ, ℓθᵤ, ∇ℓθᵤ)
    else
        #!NOTE: second θᵤ used just as a placeholder
        ℓGradientResult(θᵤ, oftype(ℓθᵤ, -Inf), θᵤ)
    end
end

function log_density(
    diff::DiffObjective,
    θᵤ::AbstractVector{T}=unconstrain_flatten(diff.objective.model, diff.objective.tagged),
) where {T<:Real}
    ℓθᵤ = _log_density(diff, θᵤ)
    if isfinite(ℓθᵤ)
        ℓDensityResult(θᵤ, T(ℓθᵤ))
    else
        ℓDensityResult(θᵤ, T(-Inf)) #oftype(ℓθᵤ, -Inf))
    end
end
function log_density_and_gradient(
    diff::DiffObjective,
    θᵤ::AbstractVector{T}=unconstrain_flatten(diff.objective.model, diff.objective.tagged),
) where {T<:Real}
    ℓθᵤ, ∇ℓθᵤ = _log_density_and_gradient(diff, θᵤ)
    if isfinite(ℓθᵤ)
        ℓGradientResult(θᵤ, ℓθᵤ, ∇ℓθᵤ)
    else
        #!NOTE: second θᵤ used just as a placeholder
        ℓGradientResult(θᵤ, oftype(ℓθᵤ, -Inf), θᵤ)
    end
end
#=
struct ℓHessianResult{T, S} <: ℓObjectiveResult
end
function log_density_and_hessian(objective::L, θᵤ::AbstractVector{T}) where {L<:AbstractDifferentiableObjective, T<:Real}
end
=#

############################################################################################
# Export
export
    ℓObjectiveResult,
    ℓGradientResult,
    ℓDensityResult,
    log_density,
    log_density_and_gradient#,
#log_density_and_hessian
