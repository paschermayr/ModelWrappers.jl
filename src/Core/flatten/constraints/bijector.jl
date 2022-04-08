############################################################################################
function construct_transform(b::Bijectors.Bijector, val)
    return b, Bijectors.inverse(b)
end

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    b::Bijectors.Bijector,
    val::Union{R,Array{R},AbstractArray},
) where {R<:Real}
    return typeof(b(val)) == typeof(val) ? true : false
end

############################################################################################
# Implicit constraint through density by specialization -> else just flatten
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::Bijectors.Bijector,
    x::Union{R,Array{R}},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    U<:UnflattenTypes,
    R<:Real,
}
    return construct_flatten(T, flattentype, unflattentype, x)
end

############################################################################################
"""
$(SIGNATURES)
Constrain unconstrained `θᵤ` given inverse transformer 'b⁻¹'.

# Examples
```julia
```

"""
function constrain(b⁻¹::S, θᵤ::T) where {S<:Bijectors.Bijector,T}
    return b⁻¹(θᵤ)
end

############################################################################################
"""
$(SIGNATURES)
Unconstrain constrained `θ` given transformer 'b'.

# Examples
```julia
```

"""
function unconstrain(b::S, θ::T) where {S<:Bijectors.Bijector,T}
    return b(θ)
end

############################################################################################
"""
$(SIGNATURES)
Compute log(abs(determinant(jacobian(`x`)))) for given transformer.

# Examples
```julia
```

"""
function log_abs_det_jac(b::Bijectors.Identity, θ::T) where {T}
    #!NOTE: Allow Fixed Params of arbitrary size work nice with bijectors
    return 0.0
end
function log_abs_det_jac(b::S, θ::T) where {S<:Bijectors.Bijector,T}
    #!NOTE: See Bijectors.logabsdetjac implementation for '-'
    return -Bijectors.logabsdetjac(b, θ)
end

############################################################################################
#Export
export
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
