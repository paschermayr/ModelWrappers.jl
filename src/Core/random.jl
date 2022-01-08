############################################################################################
"""
$(SIGNATURES)
Sample from density 'prior'.

# Examples
```julia
```

"""
function sample(_rng::Random.AbstractRNG, prior)
    return nothing
end
function sample(_rng::Random.AbstractRNG, prior::Distributions.Distribution)
    return rand(_rng, prior)
end
function sample(_rng::Random.AbstractRNG, priorᵥ::Vector{<:Distributions.Distribution})
    return rand.(_rng, priorᵥ)
end
function sample(_rng::Random.AbstractRNG, priorᵥ::AbstractArray)
    return map(iter -> sample(_rng, priorᵥ[iter]), eachindex(priorᵥ))
end
function sample(_rng::Random.AbstractRNG, priorᵥ::NamedTuple{names}) where {names}
    return NamedTuple{names}(Tuple(map(iter -> sample(_rng, priorᵥ[iter]), names)))
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
"""
$(SIGNATURES)
Evaluate eventual Jacobian adjustments from transformation of 'b' at 'θ'.

# Examples
```julia
```

"""
function log_abs_det_jac(b::Bijectors.Identity, θ::T) where {T}
    #!NOTE: A temporary solution to allow Fixed Params of arbitrary size work nice with bijectors
    return 0.0
end
function log_abs_det_jac(b::S, θ::T) where {S<:Bijectors.Bijector,T}
    #!NOTE: See Bijectors.logabsdetjac implementation for '-'
    return -Bijectors.logabsdetjac(b, θ)
end
function log_abs_det_jac(bᵥ::AbstractArray, θ::AbstractArray) where {T}
    return sum(map(log_abs_det_jac, bᵥ, θ))
end
function log_abs_det_jac(bᵥ::A, θ::B) where {A<:NamedTuple,B<:NamedTuple}
    return sum(map(log_abs_det_jac, bᵥ, θ))
end

############################################################################################
#export
export sample, log_prior, log_prior_with_transform, log_abs_det_jac
