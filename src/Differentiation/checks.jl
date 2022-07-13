############################################################################################
"""
$(SIGNATURES)
Check gradient computations of different backends against 'objective'.

# Examples
```julia
```

"""
function check_gradients(
    _rng::Random.AbstractRNG,
    objective::Objective,
    ADlibraries = [:ForwardDiff, :ReverseDiff, :Zygote],
    θᵤ = randn(_rng, length(objective)),
    difftune = map(backend -> DiffObjective(objective, AutomaticDiffTune(backend, objective)), ADlibraries);
    printoutput = true
)
## Compute Gradients
    ℓobjectiveresults = ℓGradientResult[]
    for iter in eachindex(difftune)
        push!(ℓobjectiveresults, log_density_and_gradient(difftune[iter], θᵤ))
        if printoutput
            println(ADlibraries[iter], " gradient call succesfull.")
        end
    end
## Check differences
    ℓobjective_diff = map(
        iter -> ℓobjectiveresults[1].ℓθᵤ - ℓobjectiveresults[iter].ℓθᵤ, eachindex(ℓobjectiveresults)
    )
    for iter in eachindex(ℓobjectiveresults)
        if printoutput
            println("Log objective result difference of ", ADlibraries[1], " against ", ADlibraries[iter], ": ", ℓobjective_diff[iter])
        end
    end

    ℓobjective_gradient_diff = map(
        iter -> sum(abs.(ℓobjectiveresults[1].∇ℓθᵤ .- ℓobjectiveresults[iter].∇ℓθᵤ)), eachindex(ℓobjectiveresults)
    )
    for iter in eachindex(ℓobjective_gradient_diff)
        if printoutput
            println("Log objective gradient difference of ", ADlibraries[1], " against ", ADlibraries[iter], ": ", ℓobjective_gradient_diff[iter])
        end
    end
## Compare against base Forward and ReverseDiff
    grad_fd = ForwardDiff.gradient(objective, θᵤ)
    grad_rd = ReverseDiff.gradient(objective, θᵤ)
    fdrd_diff = map(result ->
        (sum(abs.(result.∇ℓθᵤ .- grad_fd)), sum(abs.(result.∇ℓθᵤ .- grad_rd))), ℓobjectiveresults
    )
    for iter in eachindex(fdrd_diff)
        if printoutput
            println("Log objective gradient difference of ", ADlibraries[iter], " against Forward/Reverse call: ", fdrd_diff[iter])
        end
    end
## Return differences
    return (
        names = ADlibraries,
        difftune = difftune,
        ℓobjectiveresults = ℓobjectiveresults,
        ℓobjective_diff = ℓobjective_diff,
        ℓobjective_gradient_diff = ℓobjective_gradient_diff,
        Forward_Reverse_diff = fdrd_diff,
    )
end

############################################################################################
# Error handling
function checkfinite(θₜ::AbstractVector{T}) where {T<:Real}
    return _checkfinite(θₜ)
end
function checkfinite(result::T) where {T<:ℓObjectiveResult}
    return isfinite(result.ℓθᵤ) && _checkfinite(result.θᵤ) ? true : false
end
function checkfinite(
    result₀::T, result::T, min_Δ::Float64=min_Δ
) where {T<:ℓObjectiveResult}
    checkfinite(result) && ((result.ℓθᵤ - result₀.ℓθᵤ) > min_Δ) || return false
    return true
end
function checkfinite(
    ℓθ₀::R, ℓθ::R, result::T, min_Δ::Float64=min_Δ
) where {R<:Real,T<:ℓObjectiveResult}
    checkfinite(result) && ((ℓθ - ℓθ₀) > min_Δ) || return false
    return true
end

############################################################################################
"""
$(TYPEDEF)
Stores parameter in constrained space at which logdensity could not be evaluated.

# Fields
$(TYPEDFIELDS)
"""
struct ObjectiveError <: Exception
    #!NOTE: Remove Parametric types so error message is shorter
    msg::String
    ℓθᵤ::Real
    θ::NamedTuple
    θᵤ::AbstractVector
    function ObjectiveError(objective::Objective, ℓθᵤ::S, θᵤ::AbstractVector{T}) where {S<:Real, T<:Real}
        msg = "Internal error: leapfrog called from non-finite log density. Proposed parameter in constrained and unconstrained space:"
        θ = unflatten_constrain(objective.model, objective.tagged, θᵤ)
        new(msg, ℓθᵤ, θ, θᵤ)
    end
end

function checkfinite(objective::Objective, θᵤ::AbstractVector{T}) where {T<:Real}
    ArgCheck.@argcheck checkfinite(θᵤ) ObjectiveError(objective, NaN, θᵤ)
end
function checkfinite(objective::Objective, result::T) where {T<:ℓObjectiveResult}
    ArgCheck.@argcheck checkfinite(result) ObjectiveError(objective, result.ℓθᵤ, result.θᵤ)
end
function checkfinite(
    objective::Objective, result₀::T, result::T, min_Δ::Float64=min_Δ
) where {T<:ℓObjectiveResult}
    ArgCheck.@argcheck checkfinite(result₀, result, min_Δ) ObjectiveError(objective, result.ℓθᵤ, result.θᵤ)
end
function checkfinite(
    objective::Objective, ℓθ₀::R, ℓθ::R, result::T, min_Δ::Float64=min_Δ
) where {R<:Real,T<:ℓObjectiveResult}
    ArgCheck.@argcheck checkfinite(ℓθ₀, ℓθ, result, min_Δ) ObjectiveError(objective, result.ℓθᵤ, result.θᵤ)
end

############################################################################################
# Export
export
    check_gradients,
    checkfinite,
    ObjectiveError
