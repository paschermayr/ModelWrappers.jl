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
    "Number of trials to sample until finite logdensity is achieved."
    Ntrials::Int64
    function PriorInitialization(Ntrials::Integer)
        return new(Ntrials)
    end
end
PriorInitialization() = PriorInitialization(10)

"Use custom optimization technique for initialization."
struct OptimInitialization{T} <: AbstractInitialization
    "Optimization Method to find initial parameter."
    method::T
    function OptimInitialization(method::T) where {T}
        return new{T}(method)
    end
end
OptimInitialization() = OptimInitialization(nothing)

############################################################################################
function sample(_rng::Random.AbstractRNG, initialization::PriorInitialization, kernel, objective::Objective)
    # Set initial counter
    @unpack Ntrials = initialization
    ℓθᵤ = -Inf
    counter = 0
    θ = objective.model.val
    # Sample from prior until finite log target is obtained
    while !isfinite(ℓθᵤ) && counter <= Ntrials
        counter += 1
        θ = sample(_rng, objective.model, objective.tagged)
        ℓθᵤ = objective( unconstrain_flatten(objective.model.info, θ) )
    end
    ArgCheck.@argcheck counter <= Ntrials "Could not find initial parameter with finite log target density. Adjust intial values, prior, or increase number of intial samples."
    return θ
end

############################################################################################
function (initialization::NoInitialization)(_rng::Random.AbstractRNG, kernel, objective::Objective)
    #Check if initial parameter satisfy prior constraints
    ℓθᵤ = objective(unconstrain_flatten(objective.model, objective.tagged))
    @argcheck isfinite(ℓθᵤ) "Log target function at initial value not finite. Change initial parameter or sample from prior via PriorParameter"
    return objective.model.val
end

function (initialization::PriorInitialization)(_rng::Random.AbstractRNG, kernel, objective::Objective)
    #Sample from Prior
    objective.model.val = sample(_rng, initialization, kernel, objective)
    return objective.model.val
end

############################################################################################
# Export
export
    AbstractInitialization,
    NoInitialization,
    PriorInitialization,
    OptimInitialization
