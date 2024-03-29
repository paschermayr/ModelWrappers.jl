############################################################################################
"""
$(TYPEDEF)

Functor to calculate 'ℓfunc' and gradient at unconstrained 'θᵤ', including eventual Jacobian adjustments.

# Fields
$(TYPEDFIELDS)
"""
struct Objective{M<:ModelWrapper, D, T<:Tagged, F<:AbstractFloat} <: BaytesCore.AbstractObjective
    model::M
    data::D
    tagged::T
    temperature::F
    function Objective(
        model::M,
        data::D,
        tagged::T,
        temperature::F = model.info.reconstruct.default.output(1.0)
    ) where {M<:ModelWrapper,D,T<:Tagged,F<:AbstractFloat}
    ArgCheck.@argcheck 0.0 < temperature <= 1.0 "Temperature has to be bounded between 0.0 and 1.0"
        return new{M,D,T,F}(model, data, tagged, temperature)
    end
end
function Objective(
    model::ModelWrapper{M},
    data::D,
    temperature::F = model.info.reconstruct.default.output(1.0)
    ) where {M<:ModelName,D,F<:AbstractFloat}
    return Objective(model, data, Tagged(model), temperature)
end
function Objective(
    model::ModelWrapper{M},
    data::D,
    sym::S,
    temperature::F = model.info.reconstruct.default.output(1.0)
) where {M<:ModelName,D,S<:Union{Symbol,NTuple{k,Symbol} where k},F<:AbstractFloat}
    return Objective(model, data, Tagged(model, sym), temperature)
end

############################################################################################
# Basic functions for Model struct
length_constrained(objective::Objective) = length_constrained(objective.tagged)
length_unconstrained(objective::Objective) = length_unconstrained(objective.tagged)


paramnames(objective::Objective) = paramnames(objective.tagged)

############################################################################################
# A bunch of functions that can be used/extended for target model in Sampling process

"""
$(SIGNATURES)
Functor to call target function for Model given parameter and data. Default method to be used in Automatic Differentiation. model.arg and data are arguments so they can be declared as constant with Enzyme AD engine.

# Examples
```julia
```

"""
function (objective::Objective)(θ::NamedTuple, arg, data)
    return objective(θ)
end

"""
$(SIGNATURES)
Functor to call target function for Model given parameter and data.

# Examples
```julia
```

"""
function (objective::Objective)(θ::NamedTuple)
    return zero(objective.model.info.reconstruct.default.output)
end

"""
$(SIGNATURES)
Predict new data given model parameter and data.

# Examples
```julia
```

"""
function predict(_rng::Random.AbstractRNG, objective::Objective)
    return nothing
end

"""
$(SIGNATURES)
Generate statistics given model parameter and data.

# Examples
```julia
```

"""
function generate(_rng::Random.AbstractRNG, objective::Objective)
    return nothing
end
function generate(_rng::Random.AbstractRNG, objective::Objective, gen::BaytesCore.UpdateTrue)
    return generate(_rng, objective)
end
function generate(_rng::Random.AbstractRNG, objective::Objective, gen::BaytesCore.UpdateFalse)
    return nothing
end

"""
$(SIGNATURES)
Assign model dynamics for a given `objective`.

# Examples
```julia
```

"""
function dynamics(objective::Objective)
    return nothing
end

############################################################################################
#!NOTE: model.arg and data should be a function argument so for Enzyme.jl AD engine we can define them as Constant in the AD call
function (objective::Objective)(θᵤ::AbstractVector{T}, arg::A = objective.model.arg, data::D = objective.data) where {T<:Real, A, D}
    @unpack model, tagged, temperature = objective
    ## Convert vector θᵤ back to constrained space as NamedTuple
    #!NOTE: This allocates new NamedTuple only once - using a constrain!(buffer, ...) does not improve performance wrt allocations
    θ = unflattenAD_constrain(tagged.info, θᵤ)
    #!NOTE: There are border cases where θᵤ is still finite, but θ no longer after transformation, so have to cover this separately
    _checkfinite(θ) || return -Inf
    ## logabsdet_jac for transformations
    ℓjac = log_abs_det_jac(tagged.info, θ)
    _checkfinite(ℓjac) || return -Inf
    ## Evaluate objective
    ℓℒ = objective(merge(model.val, θ), arg, data)
    ## Return log posterior
    return temperature * (ℓℒ + ℓjac)
end

############################################################################################
# Export
export Objective, predict, generate, dynamics
