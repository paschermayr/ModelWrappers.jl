############################################################################################
"""
$(SIGNATURES)
Sample from constraint if 'prior'.

# Examples
```julia
```

"""
function sample_constraint(_rng::Random.AbstractRNG, prior, val)
    return val
end
function sample_constraint(_rng::Random.AbstractRNG, prior::Distributions.Distribution, val)
    return rand(_rng, prior)
end
function sample_constraint(_rng::Random.AbstractRNG, priorᵥ::Vector{<:Distributions.Distribution}, val::AbstractVector)
    return rand.(_rng, priorᵥ)
end
function sample_constraint(_rng::Random.AbstractRNG, priorᵥ::AbstractArray, val::AbstractArray)
    return map(iter -> sample_constraint(_rng, priorᵥ[iter], val[iter]), eachindex(priorᵥ))
end
function sample_constraint(_rng::Random.AbstractRNG, priorᵥ::NamedTuple{names}, val::NamedTuple) where {names}
    return NamedTuple{names}(Tuple(map(iter -> sample_constraint(_rng, priorᵥ[iter], val[iter]), names)))
end

############################################################################################
"""
$(SIGNATURES)
Evaluate Log density of 'prior' at 'θ'.

# Examples
```julia
```

"""
function log_prior(prior, θ::T) where {T}
    return 0.0
end
function log_prior(prior::Distributions.Distribution, θ::T) where {T}
    return Distributions.logpdf(prior, θ)
end
function log_prior(priorᵥ::Vector{<:Distributions.Distribution}, θ::AbstractVector)
    return sum(Distributions.logpdf.(priorᵥ, θ))
end
function log_prior(priorᵥ::AbstractArray, θ::AbstractArray) where {T}
    return sum(map(log_prior, priorᵥ, θ))
end
function log_prior(priorᵥ::A, θ::B) where {A<:NamedTuple,B<:NamedTuple}
    return sum(map(log_prior, priorᵥ, θ))
end

############################################################################################
"""
$(SIGNATURES)
Evaluate Log density and eventual Jacobian adjustments from transformation of 'prior' at 'θ'.

# Examples
```julia
```

"""
function log_prior_with_transform(prior, θ::T) where {T}
    return 0.0
end
function log_prior_with_transform(prior::Distributions.Distribution, θ::T) where {T}
    return Bijectors.logpdf_with_trans(prior, θ, true)
end
function log_prior_with_transform(
    priorᵥ::Vector{<:Distributions.Distribution}, θ::AbstractVector
)
    return sum(Bijectors.logpdf_with_trans.(priorᵥ, θ, true))
end
function log_prior_with_transform(priorᵥ::AbstractArray, θ::AbstractArray) where {T}
    return sum(map(log_prior_with_transform, priorᵥ, θ))
end
function log_prior_with_transform(priorᵥ::A, θ::B) where {A<:NamedTuple,B<:NamedTuple}
    return sum(map(log_prior_with_transform, priorᵥ, θ))
end

############################################################################################
#export
export sample_constraint, log_prior, log_prior_with_transform
