############################################################################################
"""
$(TYPEDEF)

Functor to calculate 'ℓfunc' and gradient at unconstrained 'θᵤ', including eventual Jacobian adjustments.

# Fields
$(TYPEDFIELDS)
"""
struct Objective{M<:ModelWrapper, D, T<:Tagged, F<:Real}
    model::M
    data::D
    tagged::T
#    temperature::BaytesCore.ValueHolder{F}
    temperature::F
    function Objective(
        model::M,
        data::D,
        tagged::T,
#        temperature::BaytesCore.ValueHolder{F} = BaytesCore.ValueHolder(model.info.flattendefault.output(1.0))
        temperature::F = model.info.flattendefault.output(1.0)
    ) where {M<:ModelWrapper,D,T<:Tagged, F}
    ArgCheck.@argcheck 0.0 < temperature <= 1.0 "Temperature has to be bounded between 0.0 and 1.0"
        return new{M,D,T,F}(model, data, tagged, temperature)
    end
end
function Objective(model::ModelWrapper{M}, data::D) where {M<:AbstractModel,D}
    return Objective(model, data, Tagged(model))
end
function Objective(
    model::ModelWrapper{M}, data::D, sym::S
) where {M<:AbstractModel,D,S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return Objective(model, data, Tagged(model, sym))
end

############################################################################################
# Basic functions for Model struct
length(objective::Objective) = length(objective.tagged)
paramnames(objective::Objective) = paramnames(objective.tagged)

############################################################################################
# A bunch of functions that can be used/extended for target model in Sampling process
"""
$(SIGNATURES)
Functor to call target function for Model given parameter and data.

# Examples
```julia
```

"""
function (objective::Objective)(θ)
    return 0.0
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

############################################################################################
function (objective::Objective)(θᵤ::AbstractVector{T}) where {T<:Real}
    @unpack model, data, tagged, temperature = objective
    ## Convert vector θᵤ back to constrained space as NamedTuple
    θ = constrain(tagged.info.b⁻¹, tagged.info.unflatten_AD(θᵤ))
    #!NOTE: There are border cases where θᵤ is still finite, but θ no longer after transformation, so have to cover this separately
    _checkfinite(θ) || return -Inf
    ## logabsdet_jac for transformations
    ℓjac = log_abs_det_jac(tagged.info.b, θ)
    _checkfinite(ℓjac) || return -Inf
    ## Evaluate objective
    ℓℒ = objective(merge(model.val, θ))
    ## Return log posterior
    return temperature * (ℓℒ + ℓjac)
end

############################################################################################
# Export
export Objective, predict, generate
