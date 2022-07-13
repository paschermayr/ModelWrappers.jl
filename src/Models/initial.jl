############################################################################################
"""
$(TYPEDEF)

Abstract method to initialize parameter for individual kernels.

# Fields
$(TYPEDFIELDS)
"""
abstract type AbstractInitialization end

"Use current model.val parameter as initial parameter"
struct NoInitialization <: AbstractInitialization end

"Sample (up to Ntrials) times from prior and check if log target distribution is finite at proposed parameter in unconstrained space."
struct PriorInitialization <: AbstractInitialization
    Ntrials::Int64
    function PriorInitialization(Ntrials::Integer)
        return new(Ntrials)
    end
end

"Use custom optimization technique for initialization."
struct OptimInitialization{T} <: AbstractInitialization
    method::T
    function OptimInitialization(method::T) where {T}
        return new{T}(method)
    end
end

############################################################################################
function (initialization::NoInitialization)(algorithm, objective::Objective)
    #Check if initial parameter satisfy prior constraints
    ℓθᵤ = objective(unconstrain_flatten(objective.model, objective.tagged))
    @argcheck isfinite(ℓθᵤ) "Log target function at initial value not finite. Change initial parameter or sample from prior via PriorParameter"
    return nothing
end

function (initialization::PriorInitialization)(algorithm, objective::Objective)
    # Set initial counter
    @unpack Ntrials = initialization
    ℓθᵤ = -Inf
    counter = 0
    # Sample from prior until finite log target is obtained
    while !isfinite(ℓθᵤ) && counter <= Ntrials
        counter += 1
        sample!(objective.model, objective.tagged)
        ℓθᵤ = objective(unconstrain_flatten(objective.model, objective.tagged))
    end
    ArgCheck.@argcheck counter <= NInitial "Could find initial parameter with finite log target density. Adjust intial values, prior, or increase number of intial samples."
    return nothing
end

############################################################################################
# Export
export AbstractInitialization, NoInitialization, PriorInitialization, OptimInitialization
